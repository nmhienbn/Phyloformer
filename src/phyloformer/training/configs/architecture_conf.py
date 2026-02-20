from argparse import ArgumentParser

def build_architecture_config(parser: ArgumentParser) -> None:
    arch_grp = parser.add_argument_group("MODEL ARCHITECTURE")
    arch_grp.add_argument(
        "--dropout", "-D", default=0.0, type=float, help="Dropout proportion"
    )
    arch_grp.add_argument(
        "--nb-blocks", "-b", default=6, type=int, help="Number of PF blocks"
    )
    arch_grp.add_argument(
        "--embed-dim", "-d", default=64, type=int, help="Number of embedding dimensions"
    )
    arch_grp.add_argument(
        "--nb-heads", "-H", default=4, type=int, help="Number of attention heads"
    )
