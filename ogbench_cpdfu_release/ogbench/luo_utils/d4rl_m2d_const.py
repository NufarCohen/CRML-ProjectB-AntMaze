## holds some constant value
## for rendering
MAZE_Large_str_maze_spec = \
    '############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############'

MAZE_Medium_str_maze_spec = \
    '########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########'

# MAZE_Umaze_str_maze_spec = '#####\\#GOO#\\###O#\\#OOO#\\#####'

MAZE_Giant_str_maze_spec = \
    "################\\#O#OOOOOO##OOOO#\\#O#O##O#O#OO##O#\\#OOO#OO#OOO#OOO#\\#O###O######O#O#\\#OOO#OOO#OOOO#O#\\###O#O#OO#O#O###\\#OOO#OO#OOO#OOO#\\#O#O#O######O#O#\\#O###OOO#OOO##O#\\#OOOOO#OOO#OOOO#\\################"

MAZE_Arena_str_maze_spec = \
    '########\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\#OOOOOO#\\########'


## check back load env
def get_str_maze_spec(og_e_name: str):
    """ Dec 20
    for ogbench maze env
    """
    e_name_low = og_e_name.lower()

    if 'large' in e_name_low:
        return MAZE_Large_str_maze_spec
    elif 'medium' in e_name_low:
        return MAZE_Medium_str_maze_spec
    elif 'giant' in e_name_low:
        return MAZE_Giant_str_maze_spec
    elif 'arena' in e_name_low:
        return MAZE_Arena_str_maze_spec
    else:
        assert False

