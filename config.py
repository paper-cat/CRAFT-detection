width = 0
height = 0
map_width = 0
map_height = 0


def init():
    """initialize Project Global Variables

    init width, height

    """
    global width, height, map_width, map_height
    width = 1024
    height = 1024

    map_width = int(width / 2)
    map_height = int(height / 2)
