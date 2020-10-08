width = 0
height = 0
map_width = 0
map_height = 0
epochs = 0
batch_size = 0


def init():
    """initialize Project Global Variables

    init width, height

    """
    global width, height, map_width, map_height, epochs, batch_size
    width = 1024
    height = 1024

    map_width = int(width / 2)
    map_height = int(height / 2)

    epochs = 10
    batch_size = 4
