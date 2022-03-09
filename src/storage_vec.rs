use bevy::render::{
    render_resource::{
        std140::{AsStd140, Std140},
        BindingResource, Buffer, BufferBinding, BufferDescriptor, BufferUsages,
    },
    renderer::RenderDevice,
};
use std::{marker::PhantomData, num::NonZeroU64};

pub struct StorageVec<T: AsStd140> {
    storage_buffer: Option<Buffer>,
    capacity: usize,
    len: usize,
    item_size: usize,
    phantom_data: PhantomData<T>,
}

impl<T: AsStd140> Default for StorageVec<T> {
    fn default() -> Self {
        Self {
            storage_buffer: None,
            capacity: 0,
            len: 0,
            item_size: (T::std140_size_static() + <T as AsStd140>::Output::ALIGNMENT - 1)
                & !(<T as AsStd140>::Output::ALIGNMENT - 1),
            phantom_data: PhantomData::default(),
        }
    }
}

impl<T: AsStd140> StorageVec<T> {
    #[inline]
    pub fn storage_buffer(&self) -> Option<&Buffer> {
        self.storage_buffer.as_ref()
    }

    #[inline]
    pub fn binding(&self) -> Option<BindingResource> {
        Some(BindingResource::Buffer(BufferBinding {
            buffer: self.storage_buffer()?,
            offset: 0,
            size: Some(NonZeroU64::new(self.item_size as u64).unwrap()),
        }))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn push(&mut self, device: &RenderDevice) -> u32 {
        let index = self.len();
        self.len += 1;
        self.reserve(self.len, device);
        (index * self.item_size) as u32
    }

    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) -> bool {
        if capacity > self.capacity {
            self.capacity = capacity;
            let size = self.item_size * capacity;
            self.storage_buffer = Some(device.create_buffer(&BufferDescriptor {
                label: None,
                size: size as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            true
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }
}
