# deeplearn
basic, modular structures for deep learning applications.
designed to be forked and modified for specific downstream tasks.
<----------------------------------------------------------------------------------------------->
files:
    m_dataset - handles pulling input/label pairs. includes support for simple input/label transformations (perhaps noise), but model-specific transformations are best placed in model.

    model - handles model structure, from convolutions to the epoch step (forward())
        init implements a basic multilayer perceptron
        forward(x) takes an input x and performs one forward step of the model (convolution, MLP, return logits)
        Some utilities are also included, to be called if needed (softmax, predict, ...)

    train - bulk of the built-in implementation. 
        init takes many optional parameters for flexibility.
        fit() is the main loop structure for model training.
        run(phase) controls the model training for a single epoch, and is the workhorse called by fit(). phase is either 'train' or 'test'.
