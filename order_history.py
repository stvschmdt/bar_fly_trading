from order import Order


class OrderHistory:
    def __init__(self, orders: dict[str, Order] = None):
        self._orders = orders if orders else {}

    @property
    def orders(self):
        # This makes orders immutable from outside this class, ensuring you can only ever add orders to the history,
        # never delete or overwrite. We should never have a set_orders method.
        return self._orders

    def add_order(self, order):
        self._orders[order.order_id] = order

    def get_order(self, order_id):
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order with ID {order_id} not found.")
        return order

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
            List of Order objects that match the provided filters
        """
        if not (symbol or start_date or end_date or order_types):
            raise ValueError("At least one filter must be provided to get order history.")

        for order_type in order_types:
            if not issubclass(order_type, Order):
                raise ValueError(f"Cannot filter order history on order type {order_type} because it is not an Order subclass.")

        filtered_orders = []
        # Sort orders by date ascending
        sorted_orders = sorted(self._orders.values(), key=lambda x: x.order_date)
        for order in sorted_orders:
            if symbol and order.symbol != symbol:
                continue
            if start_date and order.order_date < start_date:
                continue
            if end_date and order.order_date > end_date:
                continue
            if order_types and type(order) not in order_types:
                continue
            filtered_orders.append(order)

        return filtered_orders
