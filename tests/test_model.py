# import torch
# from src.config import Config
# from src.model import CIFAR10NetDeeper

# def test_output_shape():
#     # create model
#     # create fake batch x = torch.randn(4, 3, 32, 32)
#     # assert out.shape == (4, 10)

# def test_different_batch_sizes():
#     # test with batch sizes 1, 32, 128
#     # hint: use @pytest.mark.parametrize

# def test_output_is_logits():
#     # softmax(out).sum(dim=1) ≈ 1.0  (probabilities sum to 1)
#     # but out.sum(dim=1) should NOT be 1.0
#     # hint: torch.softmax(out, dim=1)

# def test_model_has_parameters():
#     # assert len(list(model.parameters())) > 0

# def test_wrong_input_raises():
#     # feed wrong channels (1 instead of 3)
#     # assert raises RuntimeError
#     # hint: pytest.raises