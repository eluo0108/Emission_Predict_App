# ---------------------------
# SAMPLE DATASET
# ---------------------------
# Each product is represented as:
# [price, quantity], label (type of makeup)
makeup_data = [
    ([20, 5], 'lipstick'),
    ([25, 6], 'lipstick'),
    ([30, 30], 'foundation'),
    ([32, 35], 'foundation'),
    ([15, 8], 'lipstick'),
    ([40, 10], 'mascara'),
    ([42, 12], 'mascara')
]

#The formula is: square root - ((x1 - x2)^2 + (y1 - y2)^2)

def distance(p1, p2):
    total = 0
    for i in range(len(p1)):
        diff = p1[i] - p2[i]
        total += diff ** 2
    result = total ** 0.5
    return result

def knn(data, new_point, k):
    distances = [] #This will store (distance, label) paringns from our data
    for makeup in makeup_data:
        productInfo = makeup[0]
        productName = makeup[1]
        d = distance(productInfo, new_point)
        distances.append((d,productName))

    distances.sort() #sort based on lowest distance to highest and product name

    k_labels = []
    for i in range(k):
        label = distances[i][1]
        k_labels.append(label)

    vote_counts = {}
    for label in k_labels:
        if label in vote_counts:
            vote_counts[label] += 1
        else:
            vote_counts[label] = 1

    max_votes = 0
    predicted_label = None
    for label in vote_counts:
        if vote_counts[label] > max_votes:
            max_votes = vote_counts[label]
            predicted_label = label

    return predicted_label

print(f"Prediction for 28 dollars and 32ml product {knn(makeup_data, [28,32],3)}")
