"""
this script using for clean Chinese corpus.
you can set level for clean, i.e.:
level='all', will clean all character that not Chinese, include punctuations
level='normal', this will generate corpus like normal use, reserve alphabets and numbers
level='clean', this will remove all except Chinese and Chinese punctuations

besides, if you want remove complex Chinese characters, just set this to be true:
simple_only=True
"""
import numpy as np
import os
import string

cn_punctuation_set = ['，','。','！','？','“','”','、']
en_punctuation_set = [',','.','?','!','"','"']

# corpus文集 文献 语料库
def clean_cn_corpus(file_name, clean_level='all', simple_only=True, is_save=True):
    """
    clean chinese corpus
    :param file_name:
    :param clean_level:
    :param simple_only:
    :param is_save:
    :return:
    """
    if os.path.dirname(file_name):
        base_dir = os.path.dirname(file_name)
    else:
        print('not set dir . pleace check')

    save_file = os.path.join(base_dir, os.path.basename(file_name).split('.')[0] + '_cleaned.txt')
    with open(file_name, 'r+') as f:
        clean_content = []
        for l in f.readlines():
            l = l.strip()
            if l == '':
                pass
            else:
                l = list(l)
                should_remove_words = []
                for w in l:
                    if not should_reserve(w, clean_level):
                        should_remove_words.append(w)
                clean_line = [c for c in l if c not in should_remove_words]
                clean_line = ''.join(clean_line)
                if clean_line != '':
                    clean_content.append(clean_line)
    if is_save:
        with open(save_file, 'w+') as f:
            for l in clean_content:
                f.write(l + '\n')
        print('[INFO] cleaned file have been saved to %s.' % save_file)
    return clean_content




def should_reserve(w, clean_level):
    if w == '':
        return True
    else:
        if clean_level == 'all':
            # only reserve chinese characters
            # string.punctuation输出是所有的标点符号
            if w in cn_punctuation_set or w in string.punctuation or is_alphabet(w):
                return False
            else:
                return is_chinese(w)
        elif clean_level == 'normal':
            # reserve chinese characters , English alphabet , number
            if is_chinese(w) or is_alphabet(w) or is_number(w):
                return True
            elif w in cn_punctuation_set or w in en_punctuation_set:
                return True
            else:
                return False
        elif clean_level == 'clean':
            if is_chinese(w):
                return True
            elif w in cn_punctuation_set:
                return True
            else:
                return False
        else:
            raise Exception('clean_level not support %s, please set for all, normal, clean' % clean_level)

# 判断是是中文字符？ 判断方法：查看字符的unicode编码
def is_chinese(uchar):
    """is chinese"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False

# 判断一个unicode是否是数字
def is_number(uchar):
    """is number"""
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False

# 判断是否是英文字母 判断方法：同上
def is_alphabet(uchar):
    """is alphabet 英文字母"""
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False

# 全角和半角是针对中文输入法来说的，全角一个字符占两个字符，而半角只占一个字符。
def semi_angle_to_sbc(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    # 如果不是全角字符就直接返回原来字符。 备注：上面ord('a') = 97 而 chr(97) = 'a'
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    # 判断是否是空格键
    if inside_code == 0x0020:
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
        # 除了上面空格，其他的全角半角的转换公式为：半角 = 全角 - oxfee0
    return chr(inside_code)

def sbc_to_semi_angle(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    return chr(inside_code)

def stringQ2B(ustring):
    """把字符串全角转为半角"""
    return ''.join([sbc_to_semi_angle(uchar) for uchar in ustring])

def uniform(ustring):
    """格式化字符串，完成全角转半角，大写转小写的工作"""
    return sbc_to_semi_angle(ustring).lower()

def is_other(uchar):
    """判断是否汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def string2List(ustring):
    """将ustring按照中文，字母，数字分开"""
    retList = []
    utmp= []
    for uchar in ustring:
        if is_other(uchar):
            if len(utmp) == 0:
                continue
            else:
                retList.append(''.join(utmp))
                utmp = []
        else:
            utmp.append(uchar)
    if len(utmp) != 0:
        retList.append(''.join(utmp))
    return retList

if __name__ == '__main__':
    pass
