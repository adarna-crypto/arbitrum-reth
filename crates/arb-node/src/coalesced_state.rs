//! State provider wrapper that pre-coalesces the in-memory block storage
//! into a single hashmap keyed by `(address, slot)`, so each SLOAD is one
//! lookup instead of a linear walk over every unflushed in-memory block.

use std::sync::Arc;

use alloy_primitives::{Address, BlockNumber, Bytes, StorageKey, StorageValue, B256, U256};
use reth_chain_state::BlockState;
use reth_primitives_traits::{Account, Bytecode, NodePrimitives};
use reth_storage_api::{
    errors::provider::ProviderResult, AccountReader, BlockHashReader, BytecodeReader,
    HashedPostStateProvider, StateProofProvider, StateProvider, StateProviderBox,
    StateRootProvider, StorageRootProvider,
};
use reth_trie::{
    updates::TrieUpdates, AccountProof, ExecutionWitnessMode, HashedPostState, HashedStorage,
    MultiProof, MultiProofTargets, StorageMultiProof, TrieInput,
};
use revm_database::BundleState;
use rustc_hash::{FxHashMap, FxHashSet};

/// Coalesced view of the in-memory block storage overlay.
///
/// Every `(address, slot)` explicit write from any unflushed block is
/// collapsed to its newest value. Accounts whose newest state has
/// "storage known" semantics (newly created or destroyed, matching
/// `BundleAccount::storage_slot`) are tracked separately so slot reads
/// that miss the explicit map still return `Some(U256::ZERO)` for them.
#[derive(Default, Clone)]
pub struct CoalescedOverlay {
    /// Explicit `(address, slot) -> value` from the newest block that
    /// touched each pair, after wipes have been applied.
    slots: FxHashMap<(Address, B256), U256>,
    /// Addresses whose newest state is "storage known" with no subsequent
    /// explicit write. Reads for slots not in `slots` return zero.
    wiped: FxHashSet<Address>,
}

impl CoalescedOverlay {
    /// Builds the coalesced overlay by iterating the in-memory chain from
    /// oldest to newest. A wipe (status == storage_known) for an account
    /// drops all older explicit entries for that address; later explicit
    /// writes reinstate specific slots.
    pub fn from_chain<N: NodePrimitives>(head: &BlockState<N>) -> Self {
        let mut overlay = Self::default();
        // `BlockState::chain` yields newest-first; apply oldest-first so
        // newer writes overwrite older ones.
        let blocks: Vec<&BlockState<N>> = head.chain().collect();
        for block_state in blocks.into_iter().rev() {
            overlay.extend_with_block(&block_state.block_ref().execution_output.state);
        }
        overlay
    }

    /// Folds one block's bundle into the overlay using the same
    /// wipe-then-write semantics as `from_chain`. A wipe is sticky for
    /// the account: a later non-wipe touch does not "unwipe" other slots.
    pub fn extend_with_block(&mut self, bundle: &BundleState) {
        for (addr, account) in bundle.state.iter() {
            if account.status.is_storage_known() {
                self.wiped.insert(*addr);
                self.slots.retain(|(a, _), _| a != addr);
            }
            for (slot_u256, slot_entry) in account.storage.iter() {
                let key = B256::from(*slot_u256);
                self.slots.insert((*addr, key), slot_entry.present_value);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty() && self.wiped.is_empty()
    }

    fn lookup(&self, address: Address, key: B256) -> Option<Option<U256>> {
        if let Some(&val) = self.slots.get(&(address, key)) {
            return Some(Some(val));
        }
        if self.wiped.contains(&address) {
            return Some(Some(U256::ZERO));
        }
        None
    }
}

#[derive(Clone)]
struct SharedHistory(Arc<parking_lot::Mutex<StateProviderBox>>);

impl SharedHistory {
    fn as_ref(&self) -> parking_lot::MutexGuard<'_, StateProviderBox> {
        self.0.lock()
    }
}

reth_storage_api::macros::delegate_provider_impls!(SharedHistory);

/// The complete unflushed storage index lets misses bypass the in-memory chain.
pub struct CoalescedStateProvider {
    inner: StateProviderBox,
    historical: SharedHistory,
    overlay: Arc<CoalescedOverlay>,
}

impl CoalescedStateProvider {
    /// History must describe `head.anchor()` and the overlay must cover that entire captured chain.
    pub fn new<N: NodePrimitives>(
        historical: StateProviderBox,
        head: &BlockState<N>,
        overlay: Arc<CoalescedOverlay>,
    ) -> Self {
        let historical = SharedHistory(Arc::new(parking_lot::Mutex::new(historical)));
        let inner = head.state_provider(Box::new(historical.clone())).boxed();
        Self {
            inner,
            historical,
            overlay,
        }
    }

    pub fn boxed(self) -> StateProviderBox {
        Box::new(self)
    }
}

impl BlockHashReader for CoalescedStateProvider {
    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>> {
        self.inner.block_hash(number)
    }

    fn canonical_hashes_range(
        &self,
        start: BlockNumber,
        end: BlockNumber,
    ) -> ProviderResult<Vec<B256>> {
        self.inner.canonical_hashes_range(start, end)
    }
}

impl AccountReader for CoalescedStateProvider {
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        self.inner.basic_account(address)
    }
}

impl BytecodeReader for CoalescedStateProvider {
    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        self.inner.bytecode_by_hash(code_hash)
    }
}

impl StateRootProvider for CoalescedStateProvider {
    fn state_root(&self, state: HashedPostState) -> ProviderResult<B256> {
        self.inner.state_root(state)
    }

    fn state_root_from_nodes(&self, input: TrieInput) -> ProviderResult<B256> {
        self.inner.state_root_from_nodes(input)
    }

    fn state_root_with_updates(
        &self,
        state: HashedPostState,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        self.inner.state_root_with_updates(state)
    }

    fn state_root_from_nodes_with_updates(
        &self,
        input: TrieInput,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        self.inner.state_root_from_nodes_with_updates(input)
    }
}

impl StorageRootProvider for CoalescedStateProvider {
    fn storage_root(&self, address: Address, storage: HashedStorage) -> ProviderResult<B256> {
        self.inner.storage_root(address, storage)
    }

    fn storage_proof(
        &self,
        address: Address,
        slot: B256,
        storage: HashedStorage,
    ) -> ProviderResult<reth_trie::StorageProof> {
        self.inner.storage_proof(address, slot, storage)
    }

    fn storage_multiproof(
        &self,
        address: Address,
        slots: &[B256],
        storage: HashedStorage,
    ) -> ProviderResult<StorageMultiProof> {
        self.inner.storage_multiproof(address, slots, storage)
    }
}

impl StateProofProvider for CoalescedStateProvider {
    fn proof(
        &self,
        input: TrieInput,
        address: Address,
        slots: &[B256],
    ) -> ProviderResult<AccountProof> {
        self.inner.proof(input, address, slots)
    }

    fn multiproof(
        &self,
        input: TrieInput,
        targets: MultiProofTargets,
    ) -> ProviderResult<MultiProof> {
        self.inner.multiproof(input, targets)
    }

    fn witness(
        &self,
        input: TrieInput,
        target: HashedPostState,
        mode: ExecutionWitnessMode,
    ) -> ProviderResult<Vec<Bytes>> {
        self.inner.witness(input, target, mode)
    }
}

impl HashedPostStateProvider for CoalescedStateProvider {
    fn hashed_post_state(&self, bundle_state: &BundleState) -> HashedPostState {
        self.inner.hashed_post_state(bundle_state)
    }
}

impl StateProvider for CoalescedStateProvider {
    fn storage(
        &self,
        address: Address,
        storage_key: StorageKey,
    ) -> ProviderResult<Option<StorageValue>> {
        if let Some(val) = self.overlay.lookup(address, storage_key) {
            return Ok(val);
        }
        self.historical.storage(address, storage_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_consensus::{Block, Header};
    use arb_primitives::ArbPrimitives;
    use reth_chain_state::ExecutedBlock;
    use reth_primitives_traits::RecoveredBlock;
    use reth_storage_api::noop::NoopProvider;
    use revm::state::AccountInfo;
    use revm_database::{states::plain_account::StorageSlot, AccountStatus, BundleAccount};

    fn set_account(
        bundle: &mut BundleState,
        address: u8,
        status: AccountStatus,
        slots: &[(u64, u64)],
    ) {
        let info = if matches!(
            status,
            AccountStatus::Destroyed | AccountStatus::LoadedNotExisting
        ) {
            None
        } else {
            Some(AccountInfo {
                nonce: u64::from(address),
                ..Default::default()
            })
        };
        bundle.state.insert(
            Address::repeat_byte(address),
            BundleAccount {
                info,
                original_info: None,
                storage: slots
                    .iter()
                    .map(|(key, value)| {
                        (
                            U256::from(*key),
                            StorageSlot::new_changed(U256::ZERO, U256::from(*value)),
                        )
                    })
                    .collect(),
                status,
            },
        );
    }

    fn append(
        parent: Option<Arc<BlockState<ArbPrimitives>>>,
        bundle: BundleState,
    ) -> Arc<BlockState<ArbPrimitives>> {
        let number = parent.as_ref().map_or(1, |parent| parent.number() + 1);
        let parent_hash = parent.as_ref().map_or(B256::ZERO, |parent| parent.hash());
        let mut block = ExecutedBlock::<ArbPrimitives>::default();
        block.recovered_block = Arc::new(RecoveredBlock::new_unhashed(
            Block {
                header: Header {
                    number,
                    parent_hash,
                    ..Default::default()
                },
                body: Default::default(),
            },
            vec![],
        ));
        Arc::make_mut(&mut block.execution_output).state = bundle;
        Arc::new(BlockState::with_parent(block, parent))
    }

    fn history(bundle: &BundleState) -> StateProviderBox {
        append(None, bundle.clone())
            .state_provider(Box::new(NoopProvider::<
                reth_chainspec::ChainSpec,
                ArbPrimitives,
            >::new(reth_chainspec::MAINNET.clone())))
            .boxed()
    }

    fn assert_storage_parity(actual: &dyn StateProvider, expected: &dyn StateProvider) {
        for address in 1..=8 {
            for slot in 0..=5 {
                let address = Address::repeat_byte(address);
                let key = B256::from(U256::from(slot));
                assert_eq!(
                    actual.storage(address, key).unwrap(),
                    expected.storage(address, key).unwrap(),
                    "{address:?}/{slot}"
                );
            }
        }
    }

    #[test]
    fn storage_misses_wipes_and_held_snapshots_match_memory_provider() {
        let mut persisted = BundleState::default();
        for address in 1..=4 {
            set_account(
                &mut persisted,
                address,
                AccountStatus::Changed,
                &[(0, 70), (2, 72)],
            );
        }
        let mut first = BundleState::default();
        set_account(&mut first, 1, AccountStatus::Changed, &[(0, 0), (1, 11)]);
        set_account(&mut first, 2, AccountStatus::Destroyed, &[]);
        set_account(&mut first, 3, AccountStatus::InMemoryChange, &[(1, 33)]);
        set_account(&mut first, 5, AccountStatus::LoadedNotExisting, &[]);
        let head = append(None, first);
        let held_overlay = Arc::new(CoalescedOverlay::from_chain(&head));
        let held = CoalescedStateProvider::new(history(&persisted), &head, held_overlay.clone());
        let held_reference = head.state_provider(history(&persisted));
        assert_storage_parity(&held, &held_reference);
        assert_eq!(
            held.storage(Address::repeat_byte(1), B256::ZERO).unwrap(),
            Some(U256::ZERO)
        );
        assert_eq!(
            held.storage(Address::repeat_byte(1), B256::from(U256::from(2)))
                .unwrap(),
            Some(U256::from(72))
        );
        assert_eq!(
            held.storage(Address::repeat_byte(4), B256::ZERO).unwrap(),
            Some(U256::from(70))
        );
        assert_eq!(
            held.storage(Address::repeat_byte(6), B256::ZERO).unwrap(),
            None
        );

        let mut next = BundleState::default();
        set_account(&mut next, 1, AccountStatus::Changed, &[(1, 21)]);
        set_account(&mut next, 2, AccountStatus::DestroyedChanged, &[(3, 23)]);
        set_account(&mut next, 3, AccountStatus::Changed, &[(4, 34)]);
        let mut advanced = (*held_overlay).clone();
        advanced.extend_with_block(&next);
        let next_head = append(Some(head.clone()), next);
        let actual =
            CoalescedStateProvider::new(history(&persisted), &next_head, Arc::new(advanced));
        let rebuilt = CoalescedStateProvider::new(
            history(&persisted),
            &next_head,
            Arc::new(CoalescedOverlay::from_chain(&next_head)),
        );
        let expected = next_head.state_provider(history(&persisted));
        assert_storage_parity(&actual, &expected);
        assert_storage_parity(&rebuilt, &expected);
        assert_storage_parity(&held, &held_reference);
        assert_eq!(
            actual
                .storage(Address::repeat_byte(2), B256::from(U256::from(3)))
                .unwrap(),
            Some(U256::from(23))
        );
        assert_eq!(
            actual
                .storage(Address::repeat_byte(3), B256::from(U256::from(2)))
                .unwrap(),
            Some(U256::ZERO)
        );
        assert_eq!(
            held.storage(Address::repeat_byte(1), B256::from(U256::from(1)))
                .unwrap(),
            Some(U256::from(11))
        );

        let flushed_head = BlockState::new(next_head.block());
        assert_eq!(flushed_head.anchor().hash, head.hash());
        let after_flush = CoalescedStateProvider::new(
            head.state_provider(history(&persisted)).boxed(),
            &flushed_head,
            Arc::new(CoalescedOverlay::from_chain(&flushed_head)),
        );
        assert_storage_parity(&after_flush, &expected);
        assert_storage_parity(&held, &held_reference);
    }

    #[test]
    fn every_known_storage_status_masks_historical_slots() {
        let mut persisted = BundleState::default();
        set_account(
            &mut persisted,
            1,
            AccountStatus::Changed,
            &[(0, 99), (2, 82)],
        );
        for status in [
            AccountStatus::LoadedNotExisting,
            AccountStatus::InMemoryChange,
            AccountStatus::Destroyed,
            AccountStatus::DestroyedChanged,
            AccountStatus::DestroyedAgain,
        ] {
            let mut wiped = BundleState::default();
            set_account(&mut wiped, 1, status, &[]);
            let wiped_head = append(None, wiped);
            let mut changed = BundleState::default();
            set_account(&mut changed, 1, AccountStatus::Changed, &[(1, 7)]);
            let head = append(Some(wiped_head), changed);
            let actual = CoalescedStateProvider::new(
                history(&persisted),
                &head,
                Arc::new(CoalescedOverlay::from_chain(&head)),
            );
            let expected = head.state_provider(history(&persisted));
            assert_storage_parity(&actual, &expected);
            assert_eq!(
                actual.storage(Address::repeat_byte(1), B256::ZERO).unwrap(),
                Some(U256::ZERO),
                "{status:?}"
            );
        }
    }

    #[test]
    fn account_code_and_block_delegation_preserve_full_memory_view() {
        let old_code = revm::bytecode::Bytecode::new_legacy(Bytes::from_static(&[0x00]));
        let new_code =
            revm::bytecode::Bytecode::new_legacy(Bytes::from_static(&[0x60, 0x01, 0x00]));
        let mut persisted = BundleState::default();
        set_account(&mut persisted, 1, AccountStatus::Changed, &[(0, 70)]);
        set_account(&mut persisted, 2, AccountStatus::Changed, &[]);
        persisted
            .contracts
            .insert(old_code.hash_slow(), old_code.clone());
        let mut first = BundleState::default();
        set_account(&mut first, 1, AccountStatus::Changed, &[]);
        let info = first
            .state
            .get_mut(&Address::repeat_byte(1))
            .unwrap()
            .info
            .as_mut()
            .unwrap();
        info.nonce = 43;
        info.balance = U256::from(101);
        info.code_hash = new_code.hash_slow();
        first
            .contracts
            .insert(new_code.hash_slow(), new_code.clone());
        let head = append(None, first);
        let overlay = Arc::new(CoalescedOverlay::from_chain(&head));
        assert!(overlay.is_empty());
        let actual = CoalescedStateProvider::new(history(&persisted), &head, overlay);
        let expected = head.state_provider(history(&persisted));
        for address in 1..=3 {
            let address = Address::repeat_byte(address);
            assert_eq!(
                actual.basic_account(&address).unwrap(),
                expected.basic_account(&address).unwrap()
            );
        }
        assert_eq!(
            actual
                .basic_account(&Address::repeat_byte(1))
                .unwrap()
                .unwrap()
                .nonce,
            43
        );
        for hash in [
            old_code.hash_slow(),
            new_code.hash_slow(),
            B256::repeat_byte(0xee),
        ] {
            assert_eq!(
                actual.bytecode_by_hash(&hash).unwrap(),
                expected.bytecode_by_hash(&hash).unwrap()
            );
        }
        assert!(actual
            .bytecode_by_hash(&new_code.hash_slow())
            .unwrap()
            .is_some());
        assert_eq!(actual.block_hash(head.number()).unwrap(), Some(head.hash()));
        assert_eq!(
            actual.block_hash(999).unwrap(),
            expected.block_hash(999).unwrap()
        );
        assert_eq!(
            actual.canonical_hashes_range(0, 3).unwrap(),
            expected.canonical_hashes_range(0, 3).unwrap()
        );
        assert_storage_parity(&actual, &expected);
    }
}
