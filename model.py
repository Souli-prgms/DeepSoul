import modules


class LinearModel:
    def __init__(self, input_size, num_classes):
        self.layers = [modules.Linear(input_size, 50),
                       modules.Relu(),
                       modules.Linear(50, num_classes)]
        self.loss = modules.CrossEntropy()

    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        loss = self.loss(x, targ)
        return x, loss

    def backward(self):
        grad = self.loss.backward()
        for l in reversed(self.layers):
            grad = l.backward(out_grad=grad)

    def parameters_and_gradients(self):
        for l in self.layers:
            if bool(l.parameters()):
                yield l.parameters(), l.gradients()


class ConvModel:
    def __init__(self, input_size, num_classes):
        self.layers = [modules.Conv2d(8),
                       modules.Maxpool2d(),
                       modules.Flatten(),
                       modules.Linear(1568, num_classes)]

        self.loss = modules.CrossEntropy()

    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        loss = self.loss(x, targ)
        return x, loss

    def backward(self):
        grad = self.loss.backward()
        for l in reversed(self.layers):
            grad = l.backward(grad)

    def parameters_and_gradients(self):
        for l in self.layers:
            if bool(l.parameters()):
                yield l.parameters(), l.gradients()