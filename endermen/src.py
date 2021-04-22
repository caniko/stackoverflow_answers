"""
Solution for stackoverflow thread
https://stackoverflow.com/questions/66928860/how-to-animate-a-2d-scatter-plot-given-x-y-coordinates-and-time-with-appearing/67202647#67202647
"""

from numpy.random import RandomState, SeedSequence
from numpy.random import MT19937
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ffmpeg


RESOLUTION = (12.8, 7.2)        # * 100 pixels
NUMBER_OF_FRAMES = 900


class VideoWriter:
    # Courtesy of https://github.com/kylemcdonald/python-utils/blob/master/ffmpeg.py
    def __init__(
        self,
        filename,
        video_codec="libx265",
        fps=15,
        in_pix_fmt="rgb24",
        out_pix_fmt="yuv420p",
        input_args=None,
        output_args=None,
    ):
        self.filename = filename
        self.process = None
        self.input_args = {} if input_args is None else input_args
        self.output_args = {} if output_args is None else output_args
        self.input_args["r"] = self.input_args["framerate"] = fps
        self.input_args["pix_fmt"] = in_pix_fmt
        self.output_args["pix_fmt"] = out_pix_fmt
        self.output_args["vcodec"] = video_codec

    def add(self, frame):
        if self.process is None:
            height, width = frame.shape[:2]
            self.process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    s="{}x{}".format(width, height),
                    **self.input_args,
                )
                .filter("crop", "iw-mod(iw,2)", "ih-mod(ih,2)")
                .output(self.filename, **self.output_args)
                .global_args("-loglevel", "quiet")
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        conv = frame.astype(np.uint8).tobytes()
        self.process.stdin.write(conv)

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()


def figure_to_array(figure):
    """adapted from: https://stackoverflow.com/questions/21939658/"""
    figure.canvas.draw()
    buf = figure.canvas.tostring_rgb()
    n_cols, n_rows = figure.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(n_rows, n_cols, 3)


# Generate data for the figure
rs1 = RandomState(MT19937(SeedSequence(123456789)))

time_1 = np.round(rs1.rand(232) * NUMBER_OF_FRAMES).astype(np.int16)
time_2 = time_1 + np.round(rs1.rand(232) * (NUMBER_OF_FRAMES - time_1)).astype(np.int16)
time_3 = time_2 + np.round(rs1.rand(232) * (NUMBER_OF_FRAMES - time_2)).astype(np.int16)

loc_1_x, loc_1_y, loc_2_x, loc_2_y, loc_3_x, loc_3_y = np.round(rs1.rand(6, 232) * 100, 1)

df = pd.DataFrame({
    "loc_1_time": time_1,
    "loc_1_x": loc_1_x,
    "loc_1_y": loc_1_y,
    "loc_2_time": time_2,
    "loc_2_x": loc_2_x,
    "loc_2_y": loc_2_y,
    "loc_3_time": time_3,
    "loc_3_x": loc_3_x,
    "loc_3_y": loc_3_y,
})
"""The stack answer starts here"""
# Add extra column for disappear time
df["disappear_time"] = df["loc_3_time"] + 3

all_times = df[["loc_1_time", "loc_2_time", "loc_3_time", "disappear_time"]]
change_times = np.unique(all_times)

# Prepare ticks for plotting the figure across frames
x_values = df[["loc_1_x", "loc_2_x", "loc_3_x"]].values.flatten()
x_ticks = np.array(np.linspace(x_values.min(), x_values.max(), 6), dtype=np.uint8)

y_values = df[["loc_1_y", "loc_2_y", "loc_3_y"]].values.flatten()
y_ticks = np.array(np.round(np.linspace(y_values.min(), y_values.max(), 6)), dtype=np.uint8)

sns.set_theme(style="whitegrid")
video_writer = VideoWriter("endermen.mp4")
if 0 not in change_times:
    # Generate empty figure if no person arrive at t=0
    fig, ax = plt.subplots(figsize=RESOLUTION)
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_title("People movement. T=0")

    video_writer.add(figure_to_array(fig))

    loop_range = range(1, NUMBER_OF_FRAMES)
else:
    loop_range = range(NUMBER_OF_FRAMES)

palette = sns.color_palette("tab10")        # Returns three colors from the palette (we have three groups)
animation_data_df = pd.DataFrame(columns=["x", "y", "location", "index"])
for frame_idx in loop_range:
    if frame_idx in change_times:
        plt.close("all")
        # Get person who appears/moves/disappears
        indexes, loc_nums = np.where(all_times == frame_idx)
        loc_nums += 1

        for i, loc in zip(indexes, loc_nums):
            if loc != 4:
                x, y = df[[f"loc_{loc}_x", f"loc_{loc}_y"]].iloc[i]

            if loc == 1:            # location_1
                animation_data_df = animation_data_df.append(
                    {"x": x, "y": y, "location": loc, "index": i},
                    ignore_index=True
                )
            else:
                data_index = np.where(animation_data_df["index"] == i)[0][0]
                if loc in (2, 3):   # location_2 or 3
                    animation_data_df.loc[[data_index], :] = x, y, loc, i
                elif loc == 4:      # Disappear
                    animation_data_df.drop(data_index)

        current_palette_size = len(np.unique(animation_data_df["location"]))
        fig, ax = plt.subplots(figsize=RESOLUTION)
        sns.scatterplot(
            x="x", y="y", hue="location", data=animation_data_df, ax=ax, palette=palette[:current_palette_size]
        )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_title(f"People movement. T={frame_idx}")
    video_writer.add(figure_to_array(fig))

video_writer.close()
