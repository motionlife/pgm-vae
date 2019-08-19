"""get the baseline information of all 20 datasets from paper
 <Automatic Parameter Tying: A New Approach for Regularized Parameter Learning in Markov Networks>
 by Li Chou et al. AAAI 2018
"""

baseline = {
    'nltcs': {'vars': 16, 'train': 16181, 'valid': 2157, 'test': 3236, 'pll': 4.98, 'units': [15, 14, 13, 12]},
    'msnbc': {'vars': 17, 'train': 291326, 'valid': 38843, 'test': 58265, 'pll': 6.08},
    'kdd': {'vars': 64, 'train': 180092, 'valid': 19907, 'test': 34955, 'pll': 2.07, 'units': [50, 40, 30, 20]},
    'plants': {'vars': 69, 'train': 17412, 'valid': 2321, 'test': 3482, 'pll': 10.21},
    'audio': {'vars': 100, 'train': 15000, 'valid': 2000, 'test': 3000, 'pll': 37.03, 'units': [80, 60, 40, 30]},
    'jester': {'vars': 100, 'train': 9000, 'valid': 1000, 'test': 4116, 'pll': 49.75, 'units': [70, 50, 40, 30]},
    'netflix': {'vars': 100, 'train': 15000, 'valid': 2000, 'test': 3000, 'pll': 52.67, 'units': [80, 60, 40, 30]},
    'accidents': {'vars': 111, 'train': 12758, 'valid': 1700, 'test': 2551, 'pll': 12.69, 'units': [90, 70, 50, 30]},
    'retail': {'vars': 135, 'train': 22041, 'valid': 2938, 'test': 4408, 'pll': 10.39, 'units': [100, 70, 40, 20]},
    'pumsb_star': {'vars': 163, 'train': 12262, 'valid': 1635, 'test': 2452, 'pll': 9.79, 'units': [120, 90, 60, 40]},
    'dna': {'vars': 180, 'train': 1600, 'valid': 400, 'test': 1186, 'pll': 58.46},
    'kosarek': {'vars': 190, 'train': 33375, 'valid': 4450, 'test': 6675, 'pll': 10.17, 'units': [140, 100, 50, 25]},
    'msweb': {'vars': 294, 'train': 29441, 'valid': 3270, 'test': 5000, 'pll': 13.71},
    'book': {'vars': 500, 'train': 8700, 'valid': 1159, 'test': 1739, 'pll': 35.20},
    'tmovie': {'vars': 500, 'train': 4524, 'valid': 1002, 'test': 591, 'pll': 58.50},
    'webkb': {'vars': 839, 'train': 2803, 'valid': 558, 'test': 838, 'pll': 155.51, 'units': [400, 200, 100, 50]},
    'reuters': {'vars': 889, 'train': 6532, 'valid': 1028, 'test': 1540, 'pll': 88.55},
    '20ng': {'vars': 910, 'train': 11293, 'valid': 3764, 'test': 3764, 'pll': 160.82},
    'bbc': {'vars': 1058, 'train': 1670, 'valid': 225, 'test': 330, 'pll': 256.60},
    'ad': {'vars': 1556, 'train': 2461, 'valid': 327, 'test': 491, 'pll': 6.01},
    '50-17-8': {'vars': 289, 'train': 5000, 'valid': 2000, 'test': 2000, 'pll': 49.8696},
    'bn2o-30-20-200-2a': {'vars': 50, 'train': 5000, 'valid': 2000, 'test': 2000, 'pll': 17.369},
    'fs-07': {'vars': 1225, 'train': 5000, 'valid': 2000, 'test': 2000, 'pll': 60.0505},
    'students_03_02-0000': {'vars': 376, 'train': 5000, 'valid': 2000, 'test': 2000, 'pll': 1.4775},

}
