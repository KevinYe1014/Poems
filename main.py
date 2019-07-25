
import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='Intelligence Poem and Lyric Writer. ')

    help_ = 'you can set this value in terminal --write value can be poem or lyric.'
    parser.add_argument('--w',dest= 'write', default='poem', choices=['poem', 'lyric'], help=help_)
    help_ = 'choose to train or generate.'
    parser.add_argument('--train', dest='train', action='store_true', help=help_)
    # parser.add_argument('--no-train', dest='train', action='store_true', help=help_)
    # parser.set_defaults(train=True)
    # parser.set_defaults(a= 12) 可以快速设置默认值，也可以在前面设置过的参数在重新设置默认值。

    args_ = parser.parse_args()
    return args_

if __name__ == '__main__':
    args = parse_arg()
    if args.write == 'poem':
        from inference import tang_poems
        if args.train:
            tang_poems.main(True)
        else:
            tang_poems.main(False)
    elif args.write == 'lyric':
        from inference import song_lyrics
        print(args.train)
        if args.train:
            song_lyrics.main(True)
        else:
            song_lyrics.main(False)
    else:
        print('[INFO] write option can only be poem or lyric right now. ')