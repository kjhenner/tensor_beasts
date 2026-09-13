import torch
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text
import colorsys


def display_tensor_grid(tensor):
    # Ensure the input is a 2D tensor
    if tensor.dim() != 2:
        raise ValueError("Input must be a 2D tensor")

    # Normalize the tensor values to [0, 1] range
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val)

    # Create a color gradient function
    def get_color(value):
        hue = (1 - value) * 240 / 360  # Blue (240°) to Red (0°)
        r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 1)]
        return f"#{r:02x}{g:02x}{b:02x}"

    # Create a rich Table
    table = Table(
        show_header=False,
        padding=1,
        box=box.SQUARE,
    )

    # Add columns to the table
    for _ in range(tensor.shape[1]):
        table.add_column(justify="center", vertical="middle", no_wrap=True)

    # Add rows to the table
    for y, row in enumerate(normalized):
        print(row)
        table_row = []
        for x, value in enumerate(row):
            print(value)
            color = get_color(value.item())
            cell_value = f"{tensor[y, x].item():.2f}"
            text = Text(cell_value, style=f"on {color}")
            table_row.append(text)
        table.add_row(*table_row)

    # Display the table
    console = Console()
    console.print(table)