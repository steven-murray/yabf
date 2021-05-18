from yabf import Component, Likelihood, Parameter, ParameterVector


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


class ParameterVecLikelihood(Likelihood):
    base_parameters = ParameterVector(
        "x", fiducial=0, length=3, min=-10, max=10
    ).get_params()

    def _reduce(self, ctx, **params):
        return sum(params[f"x_{i}"] for i in range(3))

    def lnl(self, model, **params):
        return -model
