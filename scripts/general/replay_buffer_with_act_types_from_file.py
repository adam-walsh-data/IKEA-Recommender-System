import os
import pathlib
from recommenders.data_utils.preprocessing import preprocess_train_data_incl_act_rew
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Get replay buffer with action types from train file"
    )

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

    train_df = preprocess_train_data_incl_act_rew(
        dir=abs_path,
        padding_id=70852,
        state_len=10,
        incl_reward=False,
        action_to_reward_dict=None,
        pad_pos=args.pad_pos,
        action_type_name="is_buy",
        session_id_name="session_id",
        action_name="item_id",
        change_out_col_names={
            "is_end": "is_done",
            "action_type": "is_buy",
            "true_state_len": "len_state",
            "true_next_state_len": "len_next_states",
        },
    )

    # Save file to .df file in same directory as given training dataset
    # Keep same col names as paper
    train_df.to_pickle(
        os.path.join(abs_path.parent.absolute(), args.name_output + ".df")
    )

    print("\nFile was successfully created.\n")
