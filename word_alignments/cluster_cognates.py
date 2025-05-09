from Levenshtein import distance
pred = "ʃtarkən"
target = "ʃtarkə"
dist = distance(pred, target)

print(dist, len(pred), dist / len(pred))