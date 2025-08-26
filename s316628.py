# Student: s316628
import numpy as np

EPS = 1e-9
def _sanitize(y: np.ndarray) -> np.ndarray:
    return np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

def pdiv(a, b):
    den = np.where(np.abs(b) < EPS, EPS, b)
    return a / den

def plog(a):
    return np.log(np.abs(a) + EPS)

def psqrt(a):
    return np.sqrt(np.abs(a))

def pexp(a):
    return np.exp(np.clip(a, -20.0, 20.0))

def ppow(a, b):
    return np.power(np.clip(a, -50.0, 50.0), np.clip(b, -3.0, 3.0))


def f1(x: np.ndarray) -> np.ndarray:
    return _sanitize(np.sin(x[0]))


def f2(x: np.ndarray) -> np.ndarray:
    return _sanitize((pdiv(((3.3904 * x[0]) * (ppow(9.4913, (6.9304 + x[0])) * pexp(3.3904))), pdiv(plog(np.abs(psqrt(-4.7216))), 9.4913)) + (((x[1] + x[2]) + x[0]) * (ppow(pexp((4.7535 - x[0])), ((x[0] + 9.4913) + (x[1] + x[2]))) * np.abs(((9.4913 + x[0]) + plog(4.7535)))))))


def f3(x: np.ndarray) -> np.ndarray:
    return _sanitize((np.abs(-8.0045) * ((((np.sin(x[1]) - (x[1] + x[1])) + np.sin((x[1] * 0.9065))) + np.sin(((x[1] * 0.9065) * 0.9065))) + np.abs((np.tanh((x[2] * x[0])) - x[0])))))


def f4(x: np.ndarray) -> np.ndarray:
    return _sanitize(pexp(pdiv(np.cos((np.tanh(pdiv(x[1], 3.2972)) * (np.tanh(x[1]) + x[1]))), np.sin(pexp(np.tanh(psqrt(-6.9863)))))))


def f5(x: np.ndarray) -> np.ndarray:
    return _sanitize(ppow(plog(np.tanh(((5.5394 - np.tanh(x[0])) - np.sin(plog(x[0]))))), (np.cos(ppow(psqrt(pdiv(-8.6398, x[0])), psqrt(pdiv(-8.6398, x[0])))) + (np.cos(plog((-3.5072 * -2.8384))) + x[1]))))


def f6(x: np.ndarray) -> np.ndarray:
    return _sanitize(((psqrt(pdiv(psqrt(-5.8150), np.sin(np.tanh(3.3138)))) * x[1]) - ((x[0] + np.tanh(np.tanh(pexp(-5.8150)))) * np.tanh(np.abs(np.sin(pdiv(-9.5093, -4.4953)))))))


def f7(x: np.ndarray) -> np.ndarray:
    return _sanitize((pexp(((x[1] * x[0]) + psqrt(plog((x[0] - x[1]))))) + (np.abs(plog((x[0] - x[1]))) * ppow(pexp(psqrt((x[0] * x[1]))), psqrt(plog((x[0] - x[1])))))))


def f8(x: np.ndarray) -> np.ndarray:
    return _sanitize(ppow(((np.tanh(np.tanh(np.tanh(0.1487))) * x[5]) + ((np.tanh(np.tanh(-1.5219)) * np.sin(plog(x[4]))) + (np.abs(np.abs(x[5])) * x[5]))), np.abs(((psqrt(pdiv(x[4], 1.8753)) * -3.9783) + (np.abs(np.abs(2.5761)) * x[5])))))
