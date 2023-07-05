# deeplearn
basic, modular structures for deep learning applications.
designed to be forked and modified for specific downstream tasks.

---

## Files:

**m_dataset**:
- handles pulling input/label pairs. 
- supports for simple input/label transformations (perhaps noise)
- but model-specific transformations should be placed in model, not here

**model**:
- handles model structure, from convolutions to epoch step
- `__init__` implements a basic multilayer perceptron
- `forward(x)` takes an input `x` and performs one forward step of the model (convolution, MLP, return logits)
- Some utilities are also included, to be called if needed (softmax, predict, ...)

**train**:
- bulk of the built-in implementation. 
- `__init__` takes many optional parameters for flexibility.
- `fit()` is the main loop structure for model training.
- `run(phase)` controls the model training for a single epoch, and is the workhorse called by `fit()`.
- `phase` is either `'train'` or `'test'`.

**main**:
- todo. example usage below
```
# Example Usage:
model = MLP()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
dataloader_train = DataLoader(...)
dataloader_val = DataLoader(...)
metrics = Metrics(...) # metrics object, i dont know what this is
output_path = "output"
save = True
verbose = True
classifier = TestTrain(model, device, criterion, optimizer, scheduler, dataloader_train, dataloader_val, metrics, output_path, save, verbose)
classifier.fit()
```
