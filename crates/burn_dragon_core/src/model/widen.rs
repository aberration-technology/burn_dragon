use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

pub(super) fn widen_3d_last_dim_prefix<B: Backend>(
    current: Tensor<B, 3>,
    fresh: Tensor<B, 3>,
    old_last: usize,
    new_last: usize,
) -> Result<Tensor<B, 3>, String> {
    let current_shape = current.shape().dims::<3>();
    let fresh_shape = fresh.shape().dims::<3>();
    if current_shape[0] != fresh_shape[0] || current_shape[1] != fresh_shape[1] {
        return Err(format!(
            "cannot widen 3d tensor with incompatible prefix shape (current={current_shape:?}, fresh={fresh_shape:?})"
        ));
    }
    if current_shape[2] != old_last || fresh_shape[2] != new_last || old_last > new_last {
        return Err(format!(
            "cannot widen 3d tensor with incompatible widened dimension (current={current_shape:?}, fresh={fresh_shape:?}, old={old_last}, new={new_last})"
        ));
    }
    if old_last == new_last {
        return Ok(current.detach());
    }
    Ok(Tensor::cat(
        vec![
            current,
            fresh.slice([0..fresh_shape[0], 0..fresh_shape[1], old_last..new_last]),
        ],
        2,
    )
    .detach())
}

pub(super) fn widen_3d_last_dim_prefix_zero_tail<B: Backend>(
    current: Tensor<B, 3>,
    fresh: Tensor<B, 3>,
    old_last: usize,
    new_last: usize,
) -> Result<Tensor<B, 3>, String> {
    let current_shape = current.shape().dims::<3>();
    let fresh_shape = fresh.shape().dims::<3>();
    if current_shape[0] != fresh_shape[0] || current_shape[1] != fresh_shape[1] {
        return Err(format!(
            "cannot widen 3d tensor with incompatible prefix shape (current={current_shape:?}, fresh={fresh_shape:?})"
        ));
    }
    if current_shape[2] != old_last || fresh_shape[2] != new_last || old_last > new_last {
        return Err(format!(
            "cannot widen 3d tensor with incompatible widened dimension (current={current_shape:?}, fresh={fresh_shape:?}, old={old_last}, new={new_last})"
        ));
    }
    if old_last == new_last {
        return Ok(current.detach());
    }
    let device = fresh.device();
    Ok(Tensor::cat(
        vec![
            current,
            Tensor::<B, 3>::zeros(
                [
                    current_shape[0],
                    current_shape[1],
                    new_last.saturating_sub(old_last),
                ],
                &device,
            ),
        ],
        2,
    )
    .detach())
}

pub(super) fn widen_2d_last_dim_prefix<B: Backend>(
    current: Tensor<B, 2>,
    fresh: Tensor<B, 2>,
    old_last: usize,
    new_last: usize,
) -> Result<Tensor<B, 2>, String> {
    let current_shape = current.shape().dims::<2>();
    let fresh_shape = fresh.shape().dims::<2>();
    if current_shape[0] != fresh_shape[0] {
        return Err(format!(
            "cannot widen 2d tensor with incompatible prefix shape (current={current_shape:?}, fresh={fresh_shape:?})"
        ));
    }
    if current_shape[1] != old_last || fresh_shape[1] != new_last || old_last > new_last {
        return Err(format!(
            "cannot widen 2d tensor with incompatible widened dimension (current={current_shape:?}, fresh={fresh_shape:?}, old={old_last}, new={new_last})"
        ));
    }
    if old_last == new_last {
        return Ok(current.detach());
    }
    Ok(Tensor::cat(
        vec![
            current,
            fresh.slice([0..fresh_shape[0], old_last..new_last]),
        ],
        1,
    )
    .detach())
}

pub(super) fn widen_2d_last_dim_prefix_zero_tail<B: Backend>(
    current: Tensor<B, 2>,
    fresh: Tensor<B, 2>,
    old_last: usize,
    new_last: usize,
) -> Result<Tensor<B, 2>, String> {
    let current_shape = current.shape().dims::<2>();
    let fresh_shape = fresh.shape().dims::<2>();
    if current_shape[0] != fresh_shape[0] {
        return Err(format!(
            "cannot widen 2d tensor with incompatible prefix shape (current={current_shape:?}, fresh={fresh_shape:?})"
        ));
    }
    if current_shape[1] != old_last || fresh_shape[1] != new_last || old_last > new_last {
        return Err(format!(
            "cannot widen 2d tensor with incompatible widened dimension (current={current_shape:?}, fresh={fresh_shape:?}, old={old_last}, new={new_last})"
        ));
    }
    if old_last == new_last {
        return Ok(current.detach());
    }
    let device = fresh.device();
    Ok(Tensor::cat(
        vec![
            current,
            Tensor::<B, 2>::zeros(
                [current_shape[0], new_last.saturating_sub(old_last)],
                &device,
            ),
        ],
        1,
    )
    .detach())
}

pub(super) fn widen_2d_headed_row_prefix<B: Backend>(
    current: Tensor<B, 2>,
    fresh: Tensor<B, 2>,
    heads: usize,
    old_per_head: usize,
    new_per_head: usize,
) -> Result<Tensor<B, 2>, String> {
    let current_shape = current.shape().dims::<2>();
    let fresh_shape = fresh.shape().dims::<2>();
    if current_shape[1] != fresh_shape[1] {
        return Err(format!(
            "cannot widen headed 2d rows with incompatible width (current={current_shape:?}, fresh={fresh_shape:?})"
        ));
    }
    if current_shape[0] != heads.saturating_mul(old_per_head)
        || fresh_shape[0] != heads.saturating_mul(new_per_head)
        || old_per_head > new_per_head
    {
        return Err(format!(
            "cannot widen headed 2d rows with incompatible head dimensions (current={current_shape:?}, fresh={fresh_shape:?}, heads={heads}, old={old_per_head}, new={new_per_head})"
        ));
    }
    if old_per_head == new_per_head {
        return Ok(current.detach());
    }

    let mut per_head = Vec::with_capacity(heads);
    for head in 0..heads {
        let current_start = head * old_per_head;
        let fresh_start = head * new_per_head;
        per_head.push(Tensor::cat(
            vec![
                current.clone().slice([
                    current_start..current_start + old_per_head,
                    0..current_shape[1],
                ]),
                fresh.clone().slice([
                    fresh_start + old_per_head..fresh_start + new_per_head,
                    0..fresh_shape[1],
                ]),
            ],
            0,
        ));
    }
    Ok(Tensor::cat(per_head, 0).detach())
}

pub(super) fn widen_2d_headed_last_dim_prefix_zero_tail<B: Backend>(
    current: Tensor<B, 2>,
    fresh: Tensor<B, 2>,
    heads: usize,
    old_per_head: usize,
    new_per_head: usize,
) -> Result<Tensor<B, 2>, String> {
    let current_shape = current.shape().dims::<2>();
    let fresh_shape = fresh.shape().dims::<2>();
    if current_shape[0] != fresh_shape[0] {
        return Err(format!(
            "cannot widen headed 2d tensor with incompatible row shape (current={current_shape:?}, fresh={fresh_shape:?})"
        ));
    }
    if current_shape[1] != heads.saturating_mul(old_per_head)
        || fresh_shape[1] != heads.saturating_mul(new_per_head)
        || old_per_head > new_per_head
    {
        return Err(format!(
            "cannot widen headed 2d tensor with incompatible head dimensions (current={current_shape:?}, fresh={fresh_shape:?}, heads={heads}, old={old_per_head}, new={new_per_head})"
        ));
    }
    if old_per_head == new_per_head {
        return Ok(current.detach());
    }

    let device = fresh.device();
    let mut per_head = Vec::with_capacity(heads);
    for head in 0..heads {
        let current_start = head * old_per_head;
        per_head.push(Tensor::cat(
            vec![
                current.clone().slice([
                    0..current_shape[0],
                    current_start..current_start + old_per_head,
                ]),
                Tensor::<B, 2>::zeros(
                    [current_shape[0], new_per_head.saturating_sub(old_per_head)],
                    &device,
                ),
            ],
            1,
        ));
    }
    Ok(Tensor::cat(per_head, 1).detach())
}

pub(super) fn widen_1d_headed_last_dim_prefix_zero_tail<B: Backend>(
    current: Tensor<B, 1>,
    fresh: Tensor<B, 1>,
    heads: usize,
    old_per_head: usize,
    new_per_head: usize,
) -> Result<Tensor<B, 1>, String> {
    let current_shape = current.shape().dims::<1>();
    let fresh_shape = fresh.shape().dims::<1>();
    if current_shape[0] != heads.saturating_mul(old_per_head)
        || fresh_shape[0] != heads.saturating_mul(new_per_head)
        || old_per_head > new_per_head
    {
        return Err(format!(
            "cannot widen headed 1d tensor with incompatible head dimensions (current={current_shape:?}, fresh={fresh_shape:?}, heads={heads}, old={old_per_head}, new={new_per_head})"
        ));
    }
    if old_per_head == new_per_head {
        return Ok(current.detach());
    }

    let device = fresh.device();
    let mut per_head = Vec::with_capacity(heads);
    for head in 0..heads {
        let current_start = head * old_per_head;
        per_head.push(Tensor::cat(
            vec![
                current
                    .clone()
                    .slice([current_start..current_start + old_per_head]),
                Tensor::<B, 1>::zeros([new_per_head.saturating_sub(old_per_head)], &device),
            ],
            0,
        ));
    }
    Ok(Tensor::cat(per_head, 0).detach())
}
