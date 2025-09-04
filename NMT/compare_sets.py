og = {'A tree and its fruit.', 'We were, even then, proposing a legal framework that would oblige the Sugar Estates to plant their rows of cane in a format that allows inter-line cropping of food crops every year, and not just when there is new cane planted every seven or so years.', 'The other will be left.', 'The king was happy.', "Yes, that's right."}
new = ['A tree and its fruit.', 'The king was happy.', 'The other will be left.', 'We were, even then, proposing a legal framework that would oblige the Sugar Estates to plant their rows of cane in a format that allows inter-line cropping of food crops every year, and not just when there is new cane planted every seven or so years.', "Yes, that's right."]

print(" OG:", len(og))
print("NEW:", len(new))

if og == set(new):
    print("passed")
else:
    print("!!!!!!IT HAS FAILED!!!!!!")