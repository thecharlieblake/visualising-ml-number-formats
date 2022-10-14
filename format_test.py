"""Validate the implementation of floating point and int formats."""
from float_format import FloatInstance, IEEE754BinaryFormat
from int_format import TwosComplementFormat, TwosComplementInstance


def test_fp16_definition() -> None:
    """Validate floating point implementation by checking FP16 against known values.

    For reference, compares against values outlined in:
    https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    """

    def fp16(s: int, e: int, m: int) -> FloatInstance:
        return FloatInstance(IEEE754BinaryFormat(5, 10), s, e, m)

    def test_val(fp_val: FloatInstance, val: float) -> None:
        assert fp_val == val, fp_val

    test_val(fp16(0, 0b00000, 0b0000000000), 0)
    test_val(fp16(0, 0b00000, 0b0000000001), 0.00000005960464477539063)
    test_val(fp16(0, 0b00000, 0b1111111111), 0.00006097555160522461)
    test_val(fp16(0, 0b00001, 0b0000000000), 0.00006103515625)
    test_val(fp16(0, 0b01101, 0b0101010101), 0.333251953125)
    test_val(fp16(0, 0b01110, 0b1111111111), 0.99951171875)
    test_val(fp16(0, 0b01111, 0b0000000000), 1.0)
    test_val(fp16(0, 0b01111, 0b0000000001), 1.0009765625)
    test_val(fp16(0, 0b11110, 0b1111111111), 65504)
    test_val(fp16(0, 0b11111, 0b0000000000), float("inf"))
    test_val(fp16(1, 0b00000, 0b0000000000), -0)
    test_val(fp16(1, 0b10000, 0b0000000000), -2)
    test_val(fp16(1, 0b11111, 0b0000000000), float("-inf"))


def test_int8_definition() -> None:
    """Validate integer point implementation by checking INT8 against known values."""

    def int8(uint: int) -> TwosComplementInstance:
        return TwosComplementInstance(TwosComplementFormat(8), uint)

    def test_val(fp_val: TwosComplementInstance, val: int) -> None:
        assert fp_val == val, fp_val

    test_val(int8(0b00000000), 0)
    test_val(int8(0b00000001), 1)
    test_val(int8(0b00000010), 2)
    test_val(int8(0b01111111), 127)
    test_val(int8(0b10000000), -128)
    test_val(int8(0b11111111), -1)
