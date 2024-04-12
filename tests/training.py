import torch
from torch.utils.data import TensorDataset

def get_model():
    return torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.Tanh(),
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid()
    )

def get_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)

def make_data(N):
    X_train = torch.rand((N, 2))
    X_dev   = torch.rand((N >> 4, 2))

    Y_train = (X_train[:, 0] > X_train[:, 1])[:, None].to(dtype=torch.float32)
    Y_dev = (X_dev[:, 0] > X_dev[:, 1])[:, None].to(dtype=torch.float32)

    dummy_train_dataset = TensorDataset(X_train, Y_train)
    dummy_val_dataset   = TensorDataset(X_dev  , Y_dev  )

    return dummy_train_dataset, dummy_val_dataset

class Context:
    def __init__(self, module, record, data_dir) -> None:
        self.module = module
        self.record = record
        self.data_dir = data_dir

        torch.manual_seed(42)

        self.net = None
        self.trainer = None

        self.data_train, self.data_test = make_data(1000)

    def reinit(self):
        torch.manual_seed(42)

        self.net = get_model()
        self.trainer = self.module.Trainer(
            "minmax", self.net, torch.nn.BCELoss(),
            get_optimizer(self.net)
        )

def test_trainer_train_step(context: Context):
    context.reinit()

    dataloader = context.trainer.make_dataloader(
        context.data_train, shuffle_data=False, batch_size=1
    )

    context.trainer.model = context.trainer.model.to(context.trainer.device)

    old_w = [
        w[1] for w in context.net.named_parameters()
        if w[0] == '3.weight'
    ][0].clone().detach()

    context.net.train()
    for i, (batch_x, batch_y) in enumerate(dataloader):
        context.trainer.train_step(batch_x, batch_y)
        if i > 5: break

    new_w = [
        w[1] for w in context.net.named_parameters()
        if w[0] == '3.weight'
    ][0]

    assert not torch.allclose(old_w, new_w)

def test_trainer_eval_step(context: Context):
    if context.trainer is None: context.reinit()

    dataloader = context.trainer.make_dataloader(
        context.data_test, shuffle_data=False, batch_size=1,
    )

    context.trainer.model = context.trainer.model.to(context.trainer.device)

    loss_val   = context.trainer.eval_step(dataloader)
    loss_val_2 = context.trainer.eval_step(dataloader)

    assert loss_val == loss_val_2

def test_trainer_make_dataloader(context: Context):
    if context.trainer is None: context.reinit()

    dataloader = context.trainer.make_dataloader(
        context.data_train, shuffle_data=False, batch_size=1
    )
    assert len(dataloader) == len(context.data_train)
    for idx, (batch_x, batch_y) in enumerate(dataloader):
        if idx == 16: break
    data_batch_x, data_batch_y = context.data_train[idx]
    assert len(batch_x) == len(batch_y)
    assert torch.allclose(batch_x, data_batch_x)
    assert torch.allclose(batch_y, data_batch_y)

def test_trainer_implementation(context: Context):
    context.reinit()

    context.trainer.train(context.data_train, num_epochs=2, save_steps=1e6, eval_steps=1e6)