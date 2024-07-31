from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, TypeVar

from .core import (
    AbstractLazyInverseOperator,
    AbstractLinearOperator,
    HomothetyOperator,
    IdentityOperator,
)


class NoReduction(BaseException):
    """Raised when no algebraic reduction is applied."""


class AbstractRule:
    """The base class for all algebraic reduction rules."""


T = TypeVar('T', bound=AbstractRule)


class RuleRegistry(Generic[T]):
    """Registry for AbstractRules.

    The rules are automatically registered when a class is encountered by the Python interpreter.
    The registry can be iterated over to retrieve the registered rules.
    """

    def __init__(self) -> None:
        self._registry: list[T] = []

    def register(self, rule: T) -> None:
        """Register a rule.

        Args:
            rule: The rule to register.
        """
        self._registry.append(rule)

    def __iter__(self) -> Iterator[T]:
        return iter(self._registry)


class AbstractNaryRule(AbstractRule, ABC):
    """A binary algebraic rule to reduce `op1 @ op2 @ op3 @ ...`."""

    operator_class: type[AbstractLinearOperator]

    @abstractmethod
    def apply(self, operands: list[AbstractLinearOperator]) -> list[AbstractLinearOperator]:
        """Applies the algebraic rule to reduce the composition `op1 @ op2 @ op3 @ ...`."""


class AlgebraicReductionRule(AbstractNaryRule):
    """N-ary rule to apply algebraic reductions."""

    def apply(self, operands: list[AbstractLinearOperator]) -> list[AbstractLinearOperator]:
        if len(operands) < 2:
            return operands

        in_structure = operands[-1].in_structure()

        identity_rule = IdentityRule()
        homothety_rule = HomothetyRule()

        operands = identity_rule.apply(operands)
        operands = homothety_rule.apply(operands)

        index = 0
        while index < len(operands) - 1:
            left, right = operands[index], operands[index + 1]

            for rule in BINARY_RULE_REGISTRY:
                try:
                    new_ops = rule.apply(left, right)
                except NoReduction:
                    continue
                operands[index : index + 2] = new_ops

                # if the rule produces a HomothetyOperator, we deal with it first
                if any(isinstance(op, HomothetyOperator) for op in new_ops):
                    operands = homothety_rule.apply(operands)
                    index = 0
                if index > 0:
                    index -= 1
                break
            else:
                index += 1

        if len(operands) == 0:
            return [IdentityOperator(in_structure)]
        return operands


class HomothetyRule(AbstractNaryRule):
    """N-ary rule to combine HomothetyOperators in compositions.

    The resulting HomothetyOperator becomes the leftmost (rightmost) operator when the composed
    operator is wide (tall).
    """

    operator_class: type[AbstractLinearOperator] = HomothetyOperator

    def apply(self, operands: list[AbstractLinearOperator]) -> list[AbstractLinearOperator]:
        if len(operands) < 2:
            return operands

        first, *_, last = operands
        homothety_number = 0
        value = 1
        new_operands = []
        for operand in operands:
            if isinstance(operand, HomothetyOperator):
                value *= operand.value
                homothety_number += 1
            else:
                new_operands.append(operand)

        if homothety_number == 0:
            return operands

        # apply the homothety on the smallest number of elements
        apply_on_left = first.out_size() <= last.in_size()
        if homothety_number == 1:
            if apply_on_left and isinstance(first, HomothetyOperator):
                return operands
            elif not apply_on_left and isinstance(last, HomothetyOperator):
                return operands

        if apply_on_left:
            return [HomothetyOperator(value, first.out_structure())] + new_operands
        return new_operands + [HomothetyOperator(value, last.in_structure())]


class IdentityRule(AbstractNaryRule):
    """N-ary rule to discard IdentityOperator in compositions."""

    operator_class: type[AbstractLinearOperator] | tuple[type[AbstractLinearOperator, ...]] = (
        IdentityOperator
    )

    def apply(self, operands: list[AbstractLinearOperator]) -> list[AbstractLinearOperator]:
        return [_ for _ in operands if not isinstance(_, IdentityOperator)]


class AbstractBinaryRule(AbstractRule, ABC):
    """A binary algebraic rule to reduce `op1 @ op2`."""

    operator_class: type[AbstractLinearOperator]

    def __init_subclass__(cls, **kwargs) -> None:
        BINARY_RULE_REGISTRY.register(cls())

    @abstractmethod
    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        """Applies the algebraic rule to reduce the composition left @ right."""


BINARY_RULE_REGISTRY = RuleRegistry[AbstractBinaryRule]()


class InverseBinaryRule(AbstractBinaryRule):
    """Binary rule for `op.I @ op = I` and `op.I @ op = I`."""

    operator_class: type[AbstractLinearOperator] = AbstractLazyInverseOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        if isinstance(left, self.operator_class):
            if left.I is right:
                return []
        if isinstance(right, self.operator_class):
            if left is right.I:
                return []
        raise NoReduction
