from collections import defaultdict

from order import Order


def convert_to_defaultdict(orders: dict[str, dict[str, Order]]):
    if not orders:
        return defaultdict(lambda: defaultdict(Order))

    return defaultdict(lambda: defaultdict(Order),
                       {order_id: defaultdict(Order, date_order_map) for order_id, date_order_map in orders.items()})


class OrderHistory:
    def __init__(self, orders: dict[str, dict[str, Order]] = None):
        # {order_id: {order_timestamp: Order}}
        self._orders = convert_to_defaultdict(orders)

    @property
    def orders(self):
        # This makes orders immutable from outside this class, ensuring you can only ever add orders to the history,
        # never delete or overwrite. We should never have a set_orders method.
        return self._orders

    def add_order(self, order):
        self._orders[order.order_id][order.order_date] = order

    def get_order(self, order_id, order_date=None):
        """
        Get an order from the history by order_id and order_date. If order_date is not provided, a random order with the
        provided order_id will be returned. This could be helpful if the caller wants to match an order_id to a symbol.
        Args:
            order_id: ID of the order to retrieve
            order_date: Date of the order to retrieve

        Returns:
            Order object that matches the provided order_id and optional order_date
        """
        date_order_map = self._orders.get(order_id)
        if not date_order_map:
            raise ValueError(f"Order with ID {order_id} not found.")
        if order_date and order_date not in date_order_map:
            raise ValueError(f"Order with ID {order_id} and date {order_date} not found.")
        return date_order_map[order_date] if order_date else list(date_order_map.values())[0]

    def get_orders(self):
        return self._orders

    def get_filtered_order_history(self, symbol: str = None, start_date: str = None, end_date: str = None, order_types: list[type] = None):
        """
        Get a list of orders, sorted by date descending, that match the provided filters. If any filter is not provided,
        it will not be applied. e.g. if no symbol is provided, we'll return orders for all symbols.
        Args:
            symbol: Symbol to filter on
            start_date: Earliest date to include orders from (inclusive)
            end_date: Latest date to include orders from (inclusive)
            order_types: List of Order subclasses to filter on (e.g. [StockOrder, OptionOrder])

        Raises:
            ValueError: If no filters are provided
            ValueError: If order_types contains a type that is not a subclass of Order

        Returns:
            defaultdict of Order objects that match the provided filters in format {symbol: {order_id: {order_date: Order}}}
        """
        if not (symbol or start_date or end_date or order_types):
            raise ValueError("At least one filter must be provided to get order history.")

        for order_type in order_types:
            if not issubclass(order_type, Order):
                raise ValueError(f"Cannot filter order history on order type {order_type} because it is not an Order subclass.")

        filtered_orders = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for order_id, date_order_map in self._orders.items():
            for order_date, order in date_order_map.items():
                if symbol and order.symbol != symbol:
                    # All orders with the same order_id will have the same symbol, so we can break out of the inner loop.
                    break
                if start_date and order.order_date < start_date:
                    continue
                if end_date and order.order_date > end_date:
                    continue
                if order_types and type(order) not in order_types:
                    continue
                filtered_orders[order.symbol][order_id][order_date] = order

        return filtered_orders
