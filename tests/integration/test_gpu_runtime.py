import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pinned_torch_executes_on_current_gpu_architecture() -> None:
    major, minor = torch.cuda.get_device_capability()
    architecture = f"sm_{major}{minor}"
    assert architecture in torch.cuda.get_arch_list()
    tensor = torch.ones(1, device="cuda")
    assert tensor.item() == 1.0
