import numpy as np 
def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

input = "The orange is the fruit of the citrus species Citrus x sinensis in the family Rutaceae. It is also called sweet orange, to distinguish it from the related Citrus x aurantium, referred to as bitter orange. The sweet orange reproduces asexually (apomixis through nucellar embryony); varieties of sweet orange arise through mutations. The orange is a hybrid between pomelo (Citrus maxima) and mandarin (Citrus reticulata). It has genes that are ~25% pomelo and ~75% mandarin; however, it is not a simple backcrossed BC1 hybrid, but hybridized over multiple generations. The chloroplast genes, and therefore the maternal line, seem to be pomelo. The sweet orange has had its full genome sequenced. Earlier estimates of the percentage of pomelo genes varying from ~50% to 6% have been reported. Sweet oranges were mentioned in Chinese literature in 314 BC. As of 1987, orange trees were found to be the most cultivated fruit tree in the world. Orange trees are widely grown in tropical and subtropical climates for their sweet fruit. The fruit of the orange tree can be eaten fresh, or processed for its juice or fragrant rind. As of 2012, oranges accounted for approximately 70% of citrus production. In 2014, 70.9 million tonnes of oranges were grown worldwide, with Brazil producing 24% of the world total followed by China and India."
segments = input.split('ee')
length = []
segment_length = [len(segment) for segment in segments]

idx1 = np.argmax(segment_length)

print(len(segments[idx1]))
print('longest segment: %s' % segments[idx1])
for segment in segments:
  length.append(len([ii for ii in find_all(segment, 'th')]))
idx2 = np.argmax(length)
import ipdb;ipdb.set_trace()
print('maximum th times %d' % length[idx2])
cond = (idx1 == idx2)
print('%d' % cond)
