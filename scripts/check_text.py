import h5py

filepath="/mnt/brain4/datasets/Twitter/final/image/836727685153968129_837010852775673857.h5"
f = h5py.File(filepath, 'r')
start, end = [int(x) for x in "836727685153968129_837010852775673857".split("_")]

key = 836776896180342784

print(key <= end)
print(key >= start)

f[str(key)]