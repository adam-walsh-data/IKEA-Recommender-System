import os
import pathlib
from recommenders.data_utils.preprocessing import preprocess_train_data
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description="Get SMORL replay buffer from train file")

    parser.add_argument(
        "-f",
        "--filename",
        help="sampled training data file path",
        metavar="FILE",
        required=True,
    )

    parser.add_argument(
        "-p",
        "--pad_pos",
        help="either 'end' or 'beg'",
        required=True,
    )

    parser.add_argument(
        "-n",
        "--name_output",
        help="output name",
        required=True,
    )
    args = parser.parse_args()

    abs_path = pathlib.Path(args.filename)

    train_df = preprocess_train_data(
        dir=abs_path,
        padding_id=70852,
        state_len=10,
        pad_pos=args.pad_pos,
        reward_name="reward",
        session_id_name="session_id",
        action_name="item_id",
        change_out_col_names={},
    )

    # Save file to .df file in same directory as given training dataset
    # Keep same col names as paper
    train_df.to_pickle(
        os.path.join(abs_path.parent.absolute(), args.name_output + ".df")
    )

    print("\nFile was successfully created.\n")
