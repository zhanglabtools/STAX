import copy
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from adjustText import adjust_text
from matplotlib.patches import Rectangle
from matplotlib import rcParams
from PIL import Image
from matplotlib.figure import Figure
from scipy.stats import pearsonr, spearmanr
from anndata import AnnData
rcParams['font.family'] = 'Arial'


def clear_fig(fig):
    if fig:
        fig.axes[0].set_xlabel(None)
        fig.axes[0].set_ylabel(None)
        fig.tight_layout()
    else:
        pass
    return fig


def clear_crop_fig(fig, boundary=100):
    """
    fig = clear_crop_fig(sc.pl.umap(adata, color=['leiden'], legend_loc=None, title='', return_fig=True, size=30,
                                    show=False), boundary=105)
    fig.savefig('123.jpg', bbox_inches='tight', dpi=300)
    :param fig:
    :param boundary:
    :return:
    """
    if fig is None:
        raise ValueError("Figure object is required.")

    if fig.axes:
        ax = fig.axes[0]
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    fig.tight_layout()
    plt.close(fig)

    if boundary == 0:
        # print('fig')
        return fig
    else:
        # print('new fig')
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        rgba = np.roll(rgba, -1, axis=-1)

        image = Image.fromarray(rgba)
        cropped_image = image.crop((
            boundary, boundary,
            image.width - boundary, image.height - boundary
        ))

        new_fig = Figure()
        ax = new_fig.add_subplot(111)
        ax.imshow(cropped_image)
        ax.axis('off')

        return new_fig


def plot_color(color_list, fig_size=(6, 6), scatter_size=800):
    # 生成示例颜色数据（两行20种随机颜色）
    plt.figure(figsize=fig_size)
    colors = copy.deepcopy(color_list)

    # 计算行列参数
    num_points_per_row = 10
    n_rows = len(colors) // num_points_per_row  # 自动计算行数

    # 创建坐标数据
    scale_factor = 0.1  # 缩放因子，调整行间距
    x_coords = [i % num_points_per_row for i in range(len(colors))]
    y_coords = [i // num_points_per_row * scale_factor * -1 for i in range(len(colors))]

    # 绘制散点图
    plt.scatter(
        x_coords,
        y_coords,
        c=colors,
        s=scatter_size,
        edgecolor='black',
        linewidth=2
    )

    # 隐藏坐标轴
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    # 创建图例
    legend_handles = [Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(
        legend_handles,
        colors,
        ncol=n_rows,
        loc='lower center',
        fontsize=10,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.2
    )

    # 调整布局
    plt.ylim(min(y_coords) - 0.5, max(y_coords) + 0.1)
    plt.tight_layout()
    plt.show()


def boxplot(data: pd.DataFrame,
            fig_size=(4, 4),
            y_lim=(0, 1)):
    data = data
    # 计算每组数据的平均值
    means = [np.mean(group) for group in data]
    # 创建箱线图
    plt.figure(figsize=fig_size)
    plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue'))  # 绘制箱线图
    # 绘制平均值的折线图
    x_positions = range(1, len(data) + 1)  # 横坐标位置
    plt.plot(x_positions, means, marker='o', color='red', label='平均值')  # 折线图，标注平均值
    # 去除横纵坐标轴和网格线
    # plt.xticks([])  # 去除横坐标刻度
    # plt.yticks([])  # 去除纵坐标刻度
    plt.gca().spines['top'].set_visible(False)  # 去除顶部边框
    plt.gca().spines['right'].set_visible(False)  # 去除右侧边框
    # plt.gca().spines['left'].set_visible(False)  # 去除左侧边框
    # plt.gca().spines['bottom'].set_visible(False)  # 去除底部边框
    # 禁用网格线
    plt.grid(False)
    # y轴范围
    plt.ylim(*y_lim)  # 将纵坐标范围设置
    # 显示图形
    plt.savefig('./result/1_knn.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def stacked_barplot(adata,
                    x_axis,
                    y_axis,
                    x_axis_order=None,
                    y_axis_order=None,
                    colors=None,
                    fontsize=10,
                    edgecolor=None,
                    linewidth=0.5,
                    save_path=None,
                    fig_size=(5, 4),
                    ):
    # 增加 x_axis y_axis 的顺序
    if colors is None:
        colors = [
            '#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2',
            '#b5bd61', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31'
        ]
    else:
        colors = colors

    df = adata.obs[[y_axis, x_axis]]
    grouped = df.groupby([x_axis, y_axis]).size().reset_index(name="count")
    total_counts = grouped.groupby(x_axis)["count"].transform("sum")
    grouped["proportion"] = grouped["count"] / total_counts
    plot_data = grouped.pivot(index=x_axis, columns=y_axis, values="proportion").fillna(0)

    if x_axis_order is not None:
        plot_data = plot_data.loc[x_axis_order, :]
    if y_axis_order is not None:
        plot_data = plot_data.loc[:, y_axis_order]
    ax = plot_data.plot(kind="bar",
                        stacked=True,
                        figsize=fig_size,
                        color=colors[:plot_data.shape[1]],
                        width=0.5,
                        edgecolor=edgecolor,
                        linewidth=linewidth,  # black
                        )
    if len(df.loc[:, y_axis].unique()) <= 10:
        ncol = 1
    elif 20> len(df.loc[:, y_axis].unique()) >= 10:
        ncol = 2
    else:
        ncol = 3
    plt.legend(
        bbox_to_anchor=(1.05, 1.05),  # 图例位置
        loc="upper left",  # 图例在左上角对齐
        frameon=False,  # 图例框是否显示边框
        fontsize=fontsize,  # 图例字体大小
        title_fontsize=fontsize,  # 图例标题字体大小
        ncol=ncol  # 将图例分为两列
    )

    plt.xlabel("")
    plt.ylabel("")
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(f'{save_path}', dpi=300, bbox_inches="tight")
    plt.show()


def plot_cell_gene(adata,
                   gene_ids=None,
                   scanpy_color='cell_type',
                   title='',
                   legend_loc='right margin',
                   groups=None,
                   edges=False,
                   edges_width=0.,
                   size=None,
                   palette=None,
                   add_outline=False,
                   outline_width=(0.3, 0.05),
                   fontsize=10,
                   fontname='Arial',
                   color='black',
                   line_color='grey',
                   line_style='-',
                   line_width=0.3,
                   arrow_length=0.15,
                   arrow_direction=(1, 1),
                   draw_arrow=False,
                   shape='full'):
    """
    绘制UMAP图并标记指定基因
    :param adata: AnnData对象
    :param gene_ids: 要标记的基因ID列表
    :param scanpy_color: 用于着色的列名
    :param title: 图标题
    :param legend_loc: 图例位置
    :param groups: 要显示的组
    :param edges: 是否显示边
    :param edges_width: 边宽
    :param size: 边宽
    :param palette: palette
    :param add_outline: add_outline
    :param outline_width: add_outline
    :param fontsize: 字体大小
    :param fontname: 字体名称
    :param color: 文本和箭头颜色
    :param line_color: 连接线颜色
    :param line_style: 连接线样式
    :param line_width: 连接线宽度
    :param arrow_length: 箭头长度
    :param arrow_direction: 箭头方向，元组形式 (dx, dy)
    :param draw_arrow: 是否绘制箭头
    :param shape: 是否绘制箭头
    :return: 图形对象
    """
    sc.set_figure_params(dpi=300, figsize=(4, 4), frameon=False)
    umap_coordinates = adata.obsm['X_umap']
    fig = sc.pl.umap(adata,
                     color=scanpy_color,
                     title=title,
                     legend_loc=legend_loc,
                     groups=groups,
                     edges=edges,
                     edges_width=edges_width,
                     size=size,
                     palette=palette,
                     add_outline=add_outline,
                     outline_width=outline_width,
                     show=False,
                     return_fig=True)

    if gene_ids is not None:
        texts = []
        coordinates = []
        arrows = []
        for gene_id in gene_ids:
            cell_index = adata.obs_names.get_loc(gene_id)
            cell_coordinate = umap_coordinates[cell_index]
            arrow_end = (cell_coordinate[0] + arrow_direction[0] * arrow_length,
                         cell_coordinate[1] + arrow_direction[1] * arrow_length)
            # text = plt.text(arrow_end[0], arrow_end[1], gene_id, fontsize=fontsize, fontname=fontname,
            #                 color=color, ha='center', va='center')
            text = plt.text(arrow_end[0], arrow_end[1], f"$\it{{{gene_id}}}$", fontsize=fontsize, fontname=fontname,
                            color=color, ha='center', va='center')
            texts.append(text)
            coordinates.append((cell_coordinate, arrow_end))

        adjust_text(texts)

        # 绘制连接线或箭头
        for i, text in enumerate(texts):
            cell_coordinate, _ = coordinates[i]
            text_position = text.get_position()
            if draw_arrow:
                arrow = plt.arrow(cell_coordinate[0] + 0.05, cell_coordinate[1] + 0.05, 0., 0., width=0.0,
                                  head_width=0.2, head_length=0.15, fc=color, ec=color, shape=shape)
                arrows.append(arrow)
            else:
                plt.plot([cell_coordinate[0], text_position[0]], [cell_coordinate[1], text_position[1]],
                         color=line_color, linestyle=line_style, linewidth=line_width)

    return fig


def plot_correlation(x, y, method='pearsonr'):
    # 生成示例数据
    np.random.seed(0)

    # 创建DataFrame
    data = pd.DataFrame({'x': x, 'y': y})

    # 计算皮尔逊相关性和p值
    if method == 'pearsonr':
        corr, p_value = pearsonr(x, y)
    elif method == 'spearmanr':
        corr, p_value = spearmanr(x, y)
    else:
        raise NotImplementedError
    print(corr, p_value)
    # 设置字体为Arial
    plt.rcParams['font.family'] = 'Arial'

    # 使用seaborn绘制带有侧面柱状图的相关性散点图
    g = sns.jointplot(x='x', y='y', data=data, kind='scatter', marginal_kws=dict(bins=20, fill=True))

    # 添加回归线
    sns.regplot(x='x', y='y', data=data, ax=g.ax_joint, scatter=False, color='r')
    # 去掉散点图和柱状图上的网格线
    g.ax_joint.grid(False)
    g.ax_marg_x.grid(False)
    g.ax_marg_y.grid(False)

    # 去掉刻度值但保留坐标轴
    g.ax_joint.set_xticklabels([])
    g.ax_joint.set_yticklabels([])
    g.ax_marg_x.set_xticklabels([])
    g.ax_marg_x.set_yticklabels([])
    g.ax_marg_y.set_xticklabels([])
    g.ax_marg_y.set_yticklabels([])

    # 去掉x和y轴标签
    g.set_axis_labels("", "")

    # 加粗坐标轴
    g.ax_joint.spines['top'].set_linewidth(2)
    g.ax_joint.spines['right'].set_linewidth(2)
    g.ax_joint.spines['left'].set_linewidth(2)
    g.ax_joint.spines['bottom'].set_linewidth(2)

    # 在图上标注皮尔逊相关性和p值，自动放置在合适的位置
    bbox_props = dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='white', alpha=0)
    g.ax_joint.annotate(f'Pearson r: {corr:.2f}\np-value: {p_value:.2e}', xy=(0.05, 0.95), xycoords='axes fraction',
                        ha='left', va='top', fontsize=30, bbox=bbox_props)
    return g.fig


def plot3d_k3d(adata,
               marker,
               spatial_key='spatial_3d',
               marker_sub=None,
               point_size=0.2,
               opacity=0.7,
               shader='flat',
               palette=None,
               ):
    """
    ref: https://k3d-jupyter.org/reference/factory.points.html
    :param adata: adata from scanpy
    :param marker: the highlight object in adata.obs or adata.var
    :param spatial_key: spatial_key
    :param marker_sub: the highlight marker_sub in marker
    :param point_size: the size of point
    :param opacity: the opacity of point
    :param shader: the shader of point
    :param palette: the color of point
    :return:
    """
    import k3d
    from k3d.colormaps import matplotlib_color_maps
    if spatial_key not in list(adata.obsm.keys()):
        print(f'{spatial_key} not in adata.obsm!')
        raise ValueError
    # sc.pl.embedding(adata, color=marker, basis=spatial_key, projection='3d')
    assert shader in ['flat', 'dot', '3d', '3dSpecular', 'mesh']
    locations = adata.obsm[spatial_key]

    if marker in adata.var.index:
        expression = adata[:, marker].X.toarray().reshape(-1)
        plt_points = k3d.points(positions=locations,  # 点的坐标
                                # 其它参考颜色：https://k3d-jupyter.org/reference/colormaps.matplotlib_color_maps.html
                                color_map=matplotlib_color_maps.viridis,  # 颜色
                                attribute=expression,  # 表达值大小
                                color_range=[expression.min(), expression.max()],  # 表达上下限
                                point_size=point_size,  # 点的大小
                                opacity=opacity,  # 不透明度
                                shader=shader,  # 点的形状
                                )
        plot = k3d.plot()
        plot += plt_points
        plot.display()

    elif marker in adata.obs.columns:
        if not marker_sub:
            marker_type = np.sort(adata.obs.loc[:, marker].unique()).astype(np)
            if palette is None:
                pellete = {item: adata.uns[f'{marker}_colors'][i] for i, item in enumerate(marker_type)}
            else:
                pellete = {item: palette[item] for i, item in enumerate(marker_type)}
            # change color to hexadecimal
            colors = adata.obs.loc[:, marker].apply(lambda x: int(pellete[x][1:], 16))
        else:
            marker_type = np.sort(adata.obs.loc[:, marker].unique()).astype(np)
            if palette is None:
                pellete = {
                    item: adata.uns[f'{marker}_colors'][i] if item == marker_sub or item in marker_sub else '#CCCCCC'
                    for i, item in enumerate(marker_type.astype(str))}
            else:
                pellete = {item: palette[item] if item == marker_sub or item in marker_sub else '#CCCCCC'
                           for i, item in enumerate(marker_type.astype(str))}
            # change color to hexadecimal
            colors = adata.obs.loc[:, marker].apply(lambda x: int(pellete[x][1:], 16))

        plt_points = k3d.points(positions=locations,  # 点的坐标
                                colors=colors,
                                # color_map=matplotlib_color_maps.viridis, # 颜色
                                # attribute=expression,  # 表达值大小
                                # color_range=[expression.min(), expression.max()], # 表达上下限
                                point_size=point_size,  # 点的大小
                                opacity=opacity,  # 不透明度
                                shader=shader,  # 点的形状
                                )
        plot = k3d.plot()
        plot += plt_points
        plot.display()
    else:
        print('marker not in adata.var or adata.obs')
        raise ValueError
    return plot


def plot3d_plotly(adata,
                  marker,
                  spatial_key='spatial_3d',
                  marker_sub=None,
                  point_size=0.2,
                  opacity=0.7,
                  figure_size=(1200, 1000),
                  visible=True,
                  showticklabels=True,
                  titlefont=24,
                  tickfont=12,
                  ):
    """
    slow rather than k3d，but can move perspective
    ref: chatgpt
    :param adata: adata from scanpy
    :param marker: the highlight object in adata.obs or adata.var
    :param spatial_key: spatial_key
    :param marker_sub: the highlight marker_sub in marker
    :param point_size: the size of point
    :param opacity: the opacity of point
    :param figure_size: the size of figure
    :param visible: visible
    :param showticklabels: showticklabels
    :param titlefont: titlefont
    :param tickfont: tickfont
    :return:
    """
    import plotly.graph_objs as go
    from plotly.colors import hex_to_rgb
    if spatial_key not in list(adata.obsm.keys()):
        print(f'{spatial_key} not in adata.obsm!')
        raise ValueError
    # sc.pl.embedding(adata, color=marker, basis=spatial_key, projection='3d')
    locations = adata.obsm[spatial_key]

    if marker in adata.var.index:
        point_indices = adata.obs.index.tolist()
        expression = adata[:, marker].X.toarray().reshape(-1)
        meta = np.stack([point_indices, expression], axis=1)
        fig = go.Figure(data=[go.Scatter3d(
            x=locations[:, 0],
            y=locations[:, 1],
            z=locations[:, 2],
            mode='markers',
            customdata=meta,
            hovertemplate=(
                    "<b>Cell Index: </b><br>%{customdata[0]}</b><br>" +
                    "<b>Expression: </b><br>%{customdata[1]:.3f}</b><br>" +
                    # "<b>Expression:</b><br>%{customdata}</b><br>" +
                    "<b>Coordinates:</b><br>" +
                    "x: %{x:.3f}<br>" +
                    "y: %{y:.3f}<br>" +
                    "z: %{z:.3f}<br>" +
                    "<extra></extra>"  # <-- 确认这一行存在
            ),
            marker=dict(  # 对标记的设置
                size=point_size,
                color=expression,  # 颜色设置
                colorscale='Viridis',  # 选择颜色
                opacity=opacity  # 透明度
            )
        )])
        # TODO
        # fig.update_layout(
        #     xaxis=dict(range=[locations[:, 0].min(), locations[:, 0].max()]),  # x轴范围从1到5
        #     yaxis=dict(range=[locations[:, 1].min(), locations[:, 1].max()]),  # y轴范围从0到20
        #     zaxis=dict(range=[locations[:, 2].min(), locations[:, 2].max()]),  # z轴范围从0到20
        # )
        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(showticklabels=False),
        #         yaxis=dict(showticklabels=False),
        #         zaxis=dict(showticklabels=False)
        #     )
        # )
        fig.update_layout(width=figure_size[0],
                          height=figure_size[1],
                          scene=dict(
                              xaxis=dict(showticklabels=showticklabels,
                                         visible=visible,
                                         titlefont=dict(family="Arial", size=titlefont),
                                         tickfont=dict(family="Arial", size=tickfont)),
                              yaxis=dict(showticklabels=showticklabels,
                                         visible=visible,
                                         titlefont=dict(family="Arial", size=titlefont),
                                         tickfont=dict(family="Arial", size=tickfont)),
                              zaxis=dict(showticklabels=showticklabels,
                                         visible=visible,
                                         titlefont=dict(family="Arial", size=titlefont),
                                         tickfont=dict(family="Arial", size=tickfont))
                          ))
        fig.show()

    elif marker in adata.obs.columns:
        if not marker_sub:
            marker_type = np.sort(adata.obs.loc[:, marker].unique()).astype(str)
            pellete = {item: adata.uns[f'{marker}_colors'][i] for i, item in enumerate(marker_type)}
            # change color to hexadecimal
            colors = adata.obs.loc[:, marker].apply(lambda x: pellete[x])
            color_rgb = [hex_to_rgb(color) for color in colors]
        else:
            marker_type = np.sort(adata.obs.loc[:, marker].unique()).astype(str)
            pellete = {item: adata.uns[f'{marker}_colors'][i] if item == marker_sub else '#CCCCCC'
                       for i, item in enumerate(marker_type.astype(str))}
            # change color to hexadecimal
            colors = adata.obs.loc[:, marker].apply(lambda x: pellete[x])
            color_rgb = [hex_to_rgb(color) for color in colors]

        # 创建3D散点图
        point_indices = adata.obs.index.tolist()
        domains = adata.obs.loc[:, marker].to_numpy()
        meta = np.stack([point_indices, domains], axis=1)
        # meta = point_indices
        trace = go.Scatter3d(
            x=locations[:, 0],
            y=locations[:, 1],
            z=locations[:, 2],
            customdata=meta,
            hovertemplate=(
                        "<b>Cell Index: </b><br>%{customdata[0]}</b><br>" +
                        "<b>Domains: </b><br>%{customdata[1]}</b><br>" +
                        "<b>Coordinates:</b><br>" +
                        "x: %{x:.3f}<br>" +
                        "y: %{y:.3f}<br>" +
                        "z: %{z:.3f}<br>" +
                        "<extra></extra>"  # <-- 确认这一行存在
                            ),
            opacity=opacity,
            mode='markers',
            marker=dict(
                size=point_size,
                color=color_rgb,  # 使用自定义RGB颜色
            )
        )
        fig = go.Figure(data=[trace])
        fig.update_layout(width=figure_size[0],
                          height=figure_size[1],
                          scene=dict(
                              xaxis=dict(showticklabels=showticklabels,
                                         visible=visible,
                                         titlefont=dict(family="Arial", size=titlefont),
                                         tickfont=dict(family="Arial", size=tickfont)),
                              yaxis=dict(showticklabels=showticklabels,
                                         visible=visible,
                                         titlefont=dict(family="Arial", size=titlefont),
                                         tickfont=dict(family="Arial", size=tickfont)),
                              zaxis=dict(showticklabels=showticklabels,
                                         visible=visible,
                                         titlefont=dict(family="Arial", size=titlefont),
                                         tickfont=dict(family="Arial", size=tickfont))
                          ))
        fig.show()

    else:
        print('marker not in adata.var or adata.obs')
        raise ValueError


def slice_adata_3d(adata: AnnData,
                   plane_point: np.ndarray,
                   plane_normal: np.ndarray,
                   spatial_key='spatial',
                   slice_thickness=0.5) -> AnnData:
    """
    slice from AnnData。

    Args:
        adata (AnnData):
        plane_point (np.ndarray): one point
        plane_normal (np.ndarray): normal vector
        spatial_key (float): spatial_key
        slice_thickness (float): slice_thickness

    Returns:
        AnnData: sliced anndata

    Raises:
        ValueError
    """
    if spatial_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{spatial_key}'] not exist")

    spatial_coords = adata.obsm[spatial_key]

    if spatial_coords.shape[1] != 3:
        raise ValueError(f"The shape of adata.obsm['spatial'] is not correct")

    normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    distances = np.dot(spatial_coords - plane_point, normal_normalized)
    mask = (distances >= -slice_thickness / 2) & (distances <= slice_thickness / 2)
    sliced_adata = adata[mask].copy()
    if sliced_adata.n_obs == 0:
        print("The size of sliced_adata is 0")

    return sliced_adata


# TODO
# def plot_cell_gene(adata,
#                    gene_ids=None,
#                    scanpy_color='cell_type',
#                    title='',
#                    legend_loc='right margin',
#                    groups=None,
#                    edges=False,
#                    edges_width=0.,
#                    size=None,
#                    add_outline=False,
#                    fontsize=10,
#                    fontname='Arial',
#                    color='black',
#                    shape='full',
#                    ):
#     """
#     :param adata:
#     :param gene_ids:
#     :param scanpy_color:
#     :param title:
#     :param legend_loc:
#     :param groups:
#     :param edges:
#     :param edges_width:
#     :param size:
#     :param add_outline:
#     :param fontsize:
#     :param fontname:
#     :param color:
#     :param shape: 'full', 'right' and 'left', recommend full and right
#     :return:
#     """
#     sc.set_figure_params(dpi=300, figsize=(4, 4), frameon=False)
#     umap_coordinates = adata.obsm['X_umap']
#     fig = sc.pl.umap(adata,
#                      color=scanpy_color,
#                      title=title,
#                      legend_loc=legend_loc,
#                      groups=groups,
#                      edges=edges,
#                      edges_width=edges_width,
#                      size=size,
#                      add_outline=add_outline,
#                      show=False,
#                      return_fig=True,
#                      )
#     if gene_ids is not None:
#         texts = []
#         arrows = []
#         for gene_id in gene_ids:
#             cell_index = adata.obs_names.get_loc(gene_id)
#             cell_coordinate = umap_coordinates[cell_index]
#             # arrow and text
#             arrow_end = (cell_coordinate[0] + 0.2, cell_coordinate[1] + 0.2)
#             text = plt.text(arrow_end[0], arrow_end[1], gene_id, fontsize=fontsize, fontname=fontname,
#                             color=color, ha='center', va='center')
#             texts.append(text)
#             arrow = plt.arrow(cell_coordinate[0] + 0.05, cell_coordinate[1] + 0.05, 0., 0., width=0.0, head_width=0.2,
#                               head_length=0.15, fc=color, ec=color, shape=shape)
#             arrows.append(arrow)
#
#         adjust_text(texts)
#     # plt.show()
#     return fig


# def plot_lfc_p(lfc_matrix, p_value_matrix, fontsize=10):
#     figsize = (10, 8)
#     plt.rcParams['font.family'] = 'Arial'
#     plt.figure(figsize=figsize)
#     mask = lfc_matrix <= 0
#     lfc_matrix[mask] = 0
#     p_value_matrix[mask] = 0
#     import seaborn as sns
#     # 绘制热图
#     ax = sns.heatmap(
#         lfc_matrix,
#         cmap="PiYG",  # PiYG RdBu
#         center=0,
#         annot=False,
#         linewidths=0.5,  # 单元格之间的线宽
#         # cbar=False,
#     )
#
#     for i in range(p_value_matrix.shape[0]):
#         for j in range(p_value_matrix.shape[1]):
#             if 0.05 > p_value_matrix.iloc[i, j] > 0.01:
#                 ax.text(j + 0.5, i + 0.5, '*', color='black', ha='center', va='center', fontsize=fontsize)
#             elif 0.01 > p_value_matrix.iloc[i, j] > 0.001:
#                 ax.text(j + 0.5, i + 0.5, '**', color='black', ha='center', va='center', fontsize=fontsize)
#             elif 0.001 > p_value_matrix.iloc[i, j] > 0.0001:
#                 ax.text(j + 0.5, i + 0.5, '***', color='black', ha='center', va='center', fontsize=fontsize)
#             elif 0.0001 > p_value_matrix.iloc[i, j] > 0:
#                 ax.text(j + 0.5, i + 0.5, '****', color='black', ha='center', va='center', fontsize=fontsize)
#
#     plt.title("Heatmap of logFC with Significance")
#     plt.xticks(fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#     plt.show()
