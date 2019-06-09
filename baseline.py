"""get the baseline information of all 20 datasets from paper
 <Automatic Parameter Tying: A New Approach for Regularized Parameter Learning in Markov Networks>
 by Li Chou et al. AAAI 2018
"""


def baseline():
    return {
        'nltcs': {'vars': 16, 'train': 16181, 'valid': 2157, 'test': 3236, 'ppl': 4.98},
        'msnbc': {'vars': 17, 'train': 291326, 'valid': 38843, 'test': 58265, 'ppl': 6.08},
        'kdd': {'vars': 64, 'train': 180092, 'valid': 19907, 'test': 34955, 'ppl': 2.07},
        'plants': {'vars': 69, 'train': 17412, 'valid': 2321, 'test': 3482, 'ppl': 10.21},
        'audio': {'vars': 100, 'train': 15000, 'valid': 2000, 'test': 3000, 'ppl': 37.03},
        'jester': {'vars': 100, 'train': 9000, 'valid': 1000, 'test': 4116, 'ppl': 49.75},
        'netflix': {'vars': 100, 'train': 15000, 'valid': 2000, 'test': 3000, 'ppl': 52.67},
        'accidents': {'vars': 111, 'train': 12758, 'valid': 1700, 'test': 2551, 'ppl': 12.69},
        'retail': {'vars': 135, 'train': 22041, 'valid': 2938, 'test': 4408, 'ppl': 10.39},
        'pumsb_star': {'vars': 163, 'train': 12262, 'valid': 1635, 'test': 2452, 'ppl': 9.79},
        'dna': {'vars': 180, 'train': 1600, 'valid': 400, 'test': 1186, 'ppl': 58.46},
        'kosarek': {'vars': 190, 'train': 33375, 'valid': 4450, 'test': 6675, 'ppl': 10.17},
        'msweb': {'vars': 294, 'train': 29441, 'valid': 3270, 'test': 5000, 'ppl': 13.71},
        'book': {'vars': 500, 'train': 8700, 'valid': 1159, 'test': 1739, 'ppl': 35.20},
        'tmovie': {'vars': 500, 'train': 4524, 'valid': 1002, 'test': 591, 'ppl': 58.50},
        'webkb': {'vars': 839, 'train': 2803, 'valid': 558, 'test': 838, 'ppl': 155.51},
        'reuters': {'vars': 889, 'train': 6532, 'valid': 1028, 'test': 1540, 'ppl': 88.55},
        '20ng': {'vars': 910, 'train': 11293, 'valid': 3764, 'test': 3764, 'ppl': 160.82},
        'bbc': {'vars': 1058, 'train': 1670, 'valid': 225, 'test': 330, 'ppl': 256.60},
        'ad': {'vars': 1556, 'train': 2461, 'valid': 327, 'test': 491, 'ppl': 6.01},
    }
