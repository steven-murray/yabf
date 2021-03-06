from yabf import Component, Likelihood, Parameter


class SimpleComponent(Component):
    provides = ("x2",)
    base_parameters = [Parameter("x", 0, min=-10, max=10)]

    def calculate(self, ctx, **param):
        return param["x"] ** 2


class SimpleLikelihood(Likelihood):
    base_parameters = [Parameter("y", 0, min=-100, max=100)]

    def _reduce(self, ctx, **params):
        print("x2: {}, y: {}", ctx["x2"], params["y"])
        return ctx["x2"] * params["y"]

    def lnl(self, model, **params):
        return -model
