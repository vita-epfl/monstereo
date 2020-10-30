"""
Class for drawing frontal, bird-eye-view and combined figures
"""
# pylint: disable=attribute-defined-outside-init
import math
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils import pixel_to_camera, get_task_error


class Printer:
    """
    Print results on images: birds eye view and computed distance
    """
    DPI = 200
    FONTSIZE_D = 14
    FONTSIZE_BV = 24
    FONTSIZE_NUM = 22
    FONTSIZE_AX = 15
    LINEWIDTH = 8
    MARKERSIZE = 16
    ATTRIBUTES = dict(stereo={'color': 'deepskyblue',
                              'numcolor': 'darkorange',
                              'linewidth': 1},
                      mono={'color': 'red',
                            'numcolor': 'firebrick',
                            'linewidth': 2},)

    def __init__(self, image, output_path, kk, output_types, epistemic=False, z_max=30, fig_width=15):

        self.im = image
        self.kk = kk
        self.output_types = output_types
        self.epistemic = epistemic
        self.z_max = z_max  # To include ellipses in the image
        self.y_scale = 1
        self.width = self.im.size[0]
        self.height = self.im.size[1]
        self.fig_width = fig_width

        # Define the output dir
        self.output_path = output_path
        self.cmap = cm.get_cmap('jet')
        self.extensions = []

        # Define variables of the class to change for every image
        self.mpl_im0 = self.stds_ale = self.stds_epi = self.xx_gt = self.zz_gt = self.xx_pred = self.zz_pred = \
            self.dd_real = self.uv_centers = self.uv_shoulders = self.uv_kps = self.boxes = self.boxes_gt = \
            self.uv_camera = self.radius = self.auxs = None

    def _process_results(self, dic_ann):
        # Include the vectors inside the interval given by z_max
        self.stds_ale = dic_ann['stds_ale']
        self.stds_epi = dic_ann['stds_epi']
        self.gt = dic_ann['gt']  # regulate ground-truth matching
        self.xx_gt = [xx[0] for xx in dic_ann['xyz_real']]
        self.xx_pred = [xx[0] for xx in dic_ann['xyz_pred']]

        # Do not print instances outside z_max
        self.zz_gt = [xx[2] if xx[2] < self.z_max - self.stds_epi[idx] else 0
                      for idx, xx in enumerate(dic_ann['xyz_real'])]
        self.zz_pred = [xx[2] if xx[2] < self.z_max - self.stds_epi[idx] else 0
                        for idx, xx in enumerate(dic_ann['xyz_pred'])]
        self.dd_pred = dic_ann['dds_pred']
        self.dd_real = dic_ann['dds_real']
        self.uv_heads = dic_ann['uv_heads']
        self.uv_shoulders = dic_ann['uv_shoulders']
        self.boxes = dic_ann['boxes']
        self.boxes_gt = dic_ann['boxes_gt']

        self.uv_camera = (int(self.im.size[0] / 2), self.im.size[1])
        self.radius = 11 / 1600 * self.width
        if dic_ann['aux']:
            self.auxs = dic_ann['aux'] if dic_ann['aux'] else None

    def factory_axes(self):
        """Create axes for figures: front bird combined"""
        axes = []
        figures = []

        #  Initialize combined figure, resizing it for aesthetic proportions
        if 'combined' in self.output_types:
            assert 'bird' and 'front' not in self.output_types, \
                "combined figure cannot be print together with front or bird ones"

            self.y_scale = self.width / (self.height * 2)  # Defined proportion
            if self.y_scale < 0.95 or self.y_scale > 1.05:  # allows more variation without resizing
                self.im = self.im.resize((self.width, round(self.height * self.y_scale)))
            self.width = self.im.size[0]
            self.height = self.im.size[1]
            fig_width = self.fig_width + 0.6 * self.fig_width
            fig_height = self.fig_width * self.height / self.width

            # Distinguish between KITTI images and general images
            fig_ar_1 = 0.8
            width_ratio = 1.9
            self.extensions.append('.multi.png')

            fig, (ax0, ax1) = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [width_ratio, 1]},
                                           figsize=(fig_width, fig_height))
            ax1.set_aspect(fig_ar_1)
            fig.set_tight_layout(True)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

            figures.append(fig)
            assert 'front' not in self.output_types and 'bird' not in self.output_types, \
                "--combined arguments is not supported with other visualizations"

        # Initialize front figure
        elif 'front' in self.output_types:
            width = self.fig_width
            height = self.fig_width * self.height / self.width
            self.extensions.append(".front.png")
            plt.figure(0)
            fig0, ax0 = plt.subplots(1, 1, figsize=(width, height))
            fig0.set_tight_layout(True)
            figures.append(fig0)

        # Create front figure axis
        if any(xx in self.output_types for xx in ['front', 'combined']):
            ax0 = self._set_axes(ax0, axis=0)

            # divider = make_axes_locatable(ax0)
            # cax = divider.append_axes('right', size='3%', pad=0.05)
            # bar_ticks = self.z_max // 5 + 1
            # norm = matplotlib.colors.Normalize(vmin=0, vmax=self.z_max)
            # scalar_mappable = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            # scalar_mappable.set_array([])
            # plt.colorbar(scalar_mappable, ticks=np.linspace(0, self.z_max, bar_ticks),
            #              boundaries=np.arange(- 0.05, self.z_max + 0.1, .1), cax=cax, label='Z [m]')

            axes.append(ax0)
        if not axes:
            axes.append(None)

        # Initialize bird-eye-view figure
        if 'bird' in self.output_types:
            self.extensions.append(".bird.png")
            fig1, ax1 = plt.subplots(1, 1)
            fig1.set_tight_layout(True)
            figures.append(fig1)
        if any(xx in self.output_types for xx in ['bird', 'combined']):
            ax1 = self._set_axes(ax1, axis=1)  # Adding field of view
            axes.append(ax1)
        return figures, axes

    def draw(self, figures, axes, dic_out, image, show_all=False, video=False, legend=True,
             save=False, show=False):

        # Process the annotation dictionary of monoloco
        self._process_results(dic_out)

        # whether to include instances that don't match the ground-truth
        iterator = range(len(self.zz_pred)) if show_all else range(len(self.zz_gt))
        if not iterator:
            print("-" * 110 + '\n' + "! No instances detected, be sure to include file with ground-truth values or "
                                     "use the command --show_all" + '\n' + "-" * 110)

        # Draw the front figure
        number = dict(flag=~video, num=97)  # character a
        self.mpl_im0.set_data(image)
        for idx in iterator:
            if any(xx in self.output_types for xx in ['front', 'combined']) and self.zz_pred[idx] > 0:
                self._draw_front(axes[0],
                                 self.dd_pred[idx],
                                 idx,
                                 number)
                number['num'] += 1

        # Draw the bird figure
        number['num'] = 97
        for idx in iterator:
            if any(xx in self.output_types for xx in ['bird', 'combined']) and self.zz_pred[idx] > 0:

                # Draw ground truth and uncertainty
                self._draw_uncertainty(axes, idx)

                # Draw bird eye view text
                if number['flag']:
                    self._draw_text_bird(axes, idx, number['num'])
                    number['num'] += 1
        # Add the legend
        if legend:
            self._draw_legend(axes)

        # Draw, save or/and show the figures
        for idx, fig in enumerate(figures):
            fig.canvas.draw()
            if save:
                fig.savefig(self.output_path + self.extensions[idx], bbox_inches='tight', dpi=self.DPI)
            if show:
                fig.show()
            plt.close(fig)

    def _draw_front(self, ax, z, idx, number):

        mode = 'stereo' if self.auxs[idx] > 0.3 else 'mono'

        #Bbox
        w = self.boxes[idx][2] - self.boxes[idx][0]
        h = (self.boxes[idx][3] - self.boxes[idx][1]) * self.y_scale
        w_gt = self.boxes_gt[idx][2] - self.boxes_gt[idx][0]
        h_gt = (self.boxes_gt[idx][3] - self.boxes_gt[idx][1]) * self.y_scale
        x0 = self.boxes[idx][0]
        y0 = self.boxes[idx][1] * self.y_scale
        y1 = y0 + h
        rectangle = Rectangle((x0, y0),
                              width=w,
                              height=h,
                              fill=False,
                              color=self.ATTRIBUTES[mode]['color'],
                              linewidth=self.ATTRIBUTES[mode]['linewidth'])
        # rectangle_gt = Rectangle((self.boxes_gt[idx][0], self.boxes_gt[idx][1] * self.y_scale),
        #                          width=ww_box_gt, height=hh_box_gt, fill=False, color='g', linewidth=1)
        # axes[0].add_patch(rectangle_gt)
        ax.add_patch(rectangle)
        z_str = str(z).split(sep='.')
        text = z_str[0] + '.' + z_str[1][0]
        bbox_config = {'facecolor': self.ATTRIBUTES[mode]['color'], 'alpha': 0.4, 'linewidth': 0}
        ax.annotate(
            text,
            (x0-1.5, y1+24),
            fontsize=self.FONTSIZE_D,
            weight='bold',
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white',
            bbox=bbox_config,
        )
        if number['flag']:
            ax.text(x0 - 17,
                    y1 + 14,
                    chr(number['num']),
                    fontsize=self.FONTSIZE_NUM,
                    color=self.ATTRIBUTES[mode]['numcolor'],
                    weight='bold')

    def _draw_text_bird(self, axes, idx, num):
        """Plot the number in the bird eye view map"""
        mode = 'stereo' if self.auxs[idx] > 0.3 else 'mono'
        std = self.stds_epi[idx] if self.stds_epi[idx] > 0 else self.stds_ale[idx]
        theta = math.atan2(self.zz_pred[idx], self.xx_pred[idx])

        delta_x = std * math.cos(theta)
        delta_z = std * math.sin(theta)

        axes[1].text(self.xx_pred[idx] + delta_x + 0.2, self.zz_pred[idx] + delta_z + 0/2, chr(num),
                     fontsize=self.FONTSIZE_BV,
                     color=self.ATTRIBUTES[mode]['numcolor'])

    def _draw_uncertainty(self, axes, idx):

        theta = math.atan2(self.zz_pred[idx], self.xx_pred[idx])
        dic_std = {'ale': self.stds_ale[idx], 'epi': self.stds_epi[idx]}
        dic_x, dic_y = {}, {}

        # Aleatoric and epistemic
        for key, std in dic_std.items():
            delta_x = std * math.cos(theta)
            delta_z = std * math.sin(theta)
            dic_x[key] = (self.xx_pred[idx] - delta_x, self.xx_pred[idx] + delta_x)
            dic_y[key] = (self.zz_pred[idx] - delta_z, self.zz_pred[idx] + delta_z)

        # MonoLoco
        if not self.auxs:
            axes[1].plot(dic_x['epi'],
                         dic_y['epi'],
                         color='coral',
                         linewidth=round(self.LINEWIDTH/2),
                         label="Epistemic Uncertainty")

            axes[1].plot(dic_x['ale'],
                         dic_y['ale'],
                         color='deepskyblue',
                         linewidth=self.LINEWIDTH,
                         label="Aleatoric Uncertainty")

            axes[1].plot(self.xx_pred[idx],
                         self.zz_pred[idx],
                         color='cornflowerblue',
                         label="Prediction",
                         markersize=self.MARKERSIZE,
                         marker='o')

            if self.gt[idx]:
                axes[1].plot(self.xx_gt[idx],
                             self.zz_gt[idx],
                             color='k',
                             label="Ground-truth",
                             markersize=8,
                             marker='x')

        # MonStereo(stereo case)
        elif self.auxs[idx] > 0.5:
            axes[1].plot(dic_x['ale'],
                         dic_y['ale'],
                         color='r',
                         linewidth=self.LINEWIDTH,
                         label="Prediction (mono)")

            axes[1].plot(dic_x['ale'],
                         dic_y['ale'],
                         color='deepskyblue',
                         linewidth=self.LINEWIDTH,
                         label="Prediction (stereo+mono)")

            if self.gt[idx]:
                axes[1].plot(self.xx_gt[idx],
                             self.zz_gt[idx],
                             color='k',
                             label="Ground-truth",
                             markersize=self.MARKERSIZE,
                             marker='x')

        # MonStereo (monocular case)
        else:
            axes[1].plot(dic_x['ale'],
                         dic_y['ale'],
                         color='deepskyblue',
                         linewidth=self.LINEWIDTH,
                         label="Prediction (stereo+mono)")

            axes[1].plot(dic_x['ale'],
                         dic_y['ale'],
                         color='r',
                         linewidth=self.LINEWIDTH,
                         label="Prediction (mono)")
            if self.gt[idx]:
                axes[1].plot(self.xx_gt[idx],
                             self.zz_gt[idx],
                             color='k',
                             label="Ground-truth",
                             markersize=self.MARKERSIZE,
                             marker='x')

    def _draw_legend(self, axes):
        # Bird eye view legend
        if any(xx in self.output_types for xx in ['bird', 'combined']):
            handles, labels = axes[1].get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            axes[1].legend(by_label.values(), by_label.keys(), loc='best', prop={'size': 15})

    def _set_axes(self, ax, axis):
        assert axis in (0, 1)

        if axis == 0:
            ax.set_axis_off()
            ax.set_xlim(0, self.width)
            ax.set_ylim(self.height, 0)
            self.mpl_im0 = ax.imshow(self.im)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        else:
            uv_max = [0., float(self.height)]
            xyz_max = pixel_to_camera(uv_max, self.kk, self.z_max)
            x_max = abs(xyz_max[0])  # shortcut to avoid oval circles in case of different kk
            corr = round(float(x_max / 3))
            ax.plot([0, x_max], [0, self.z_max], 'k--')
            ax.plot([0, -x_max], [0, self.z_max], 'k--')
            ax.set_xlim(-x_max + corr, x_max - corr)
            ax.set_ylim(0, self.z_max + 1)
            ax.set_xlabel("X [m]")
            plt.xticks(fontsize=self.FONTSIZE_AX)
            plt.yticks(fontsize=self.FONTSIZE_AX)
        return ax

    # def _draw_ellipses(self, axes, idx):
    #     """draw uncertainty ellipses"""
    #     target = get_task_error(self.dd_real[idx])
    #     angle_gt = get_angle(self.xx_gt[idx], self.zz_gt[idx])
    #     ellipse_real = Ellipse((self.xx_gt[idx], self.zz_gt[idx]), width=target * 2, height=1,
    #                            angle=angle_gt, color='lightgreen', fill=True, label="Task error")
    #     axes[1].add_patch(ellipse_real)
    #     if abs(self.zz_gt[idx] - self.zz_pred[idx]) > 0.001:
    #         axes[1].plot(self.xx_gt[idx], self.zz_gt[idx], 'kx', label="Ground truth", markersize=3)
    #
    #     angle = get_angle(self.xx_pred[idx], self.zz_pred[idx])
    #     ellipse_ale = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_ale[idx] * 2,
    #                           height=1, angle=angle, color='b', fill=False, label="Aleatoric Uncertainty",
    #                           linewidth=1.3)
    #     ellipse_var = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_epi[idx] * 2,
    #                           height=1, angle=angle, color='r', fill=False, label="Uncertainty",
    #                           linewidth=1, linestyle='--')
    #
    #     axes[1].add_patch(ellipse_ale)
    #     if self.epistemic:
    #         axes[1].add_patch(ellipse_var)
    #
    #     axes[1].plot(self.xx_pred[idx], self.zz_pred[idx], 'ro', label="Predicted", markersize=3)

    # def draw_circle(self, axes, uv, color):
    #
    #     circle = Circle((uv[0], uv[1] * self.y_scale), radius=self.radius, color=color, fill=True)
    #     axes[0].add_patch(circle)

def get_angle(xx, zz):
    """Obtain the points to plot the confidence of each annotation"""

    theta = math.atan2(zz, xx)
    angle = theta * (180 / math.pi)

    return angle
