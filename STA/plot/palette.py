color_domain1 = [
    '#e64825',  # red
    '#f6c0cc',  # pink
    '#f18c25',  # orange
    '#faeca8',  # yellow
    '#97d1a0',  # green
    '#98f5e1',  # cyan
    '#abdafc',  # blue
    '#716ea9',  # purple
    '#c0cae1',  # black
    '#dda15e',  # brown
]

color_domain2 = [
    '#ff3300',  # red
    '#ff8fab',  # pink
    '#f48c06',  # orange
    '#ffd500',  # yellow
    '#52b788',  # green
    '#06d6a0',  # cyan
    '#00b4d8',  # blue
    '#9f86c0',  # purple
    '#62676e',  # black
    '#bc6c25',  # brown
]

color_domain3 = [
    '#dd0426',  # red
    '#fb6f92',  # pink
    '#f27059',  # orange
    '#ffea00',  # yellow
    '#40916c',  # green
    '#1f7a8c',  # cyan
    '#0077b6',  # blue
    '#5e548e',  # purple
    '#2b2d42',  # black
    '#432818',  # brown
]

color_scanpy20 = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#bcbd22',
    '#17becf',
    '#aec7e8',
    '#ffbb78',
    '#98df8a',
    '#ff9896',
    '#c5b0d5',
    '#c49c94',
    '#f7b6d2',
    '#dbdb8d',
    '#9edae5',
    '#ad494a',
    '#8c6d31',
]

color_batch1 = ['#1982c4', '#f3722c']
color_batch2 = ['#ea3546', '#90be6d']
color_batch3 = ['#fee440', '#9b5de5']
color_batch = ['#1982c4', '#f3722c', '#ea3546', '#90be6d', '#fee440', '#9b5de5']


def get_color(name='color_domain1'):
    if name == 'color_domain1':
        return color_domain1
    elif name == 'color_domain2':
        return color_domain2
    elif name == 'color_domain3':
        return color_domain3
    elif name == 'color_scanpy20':
        return color_scanpy20
    elif name == 'color_batch1':
        return color_batch1
    elif name == 'color_batch2':
        return color_batch2
    elif name == 'color_batch3':
        return color_batch3
    elif name == 'color_batch':
        return color_batch
    else:
        raise NotImplementedError
