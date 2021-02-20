def computeEditDistance(s, t):
    cache = {}
    def recurse(m, n):
        """
        Return the minimum edit distance between:
        - first m letters of s
        - first m letters of t
        """

        if (m, n) in cache:
            return cache[(m, n)]
        if m == 0:
            result = n
        elif n == 0:
            result = m
        elif s[m - 1] == t[n - 1]:
            result = recurse(m - 1, n - 1)
        else:
            subCost = 1 + recurse(m - 1, n - 1)
            insCost = 1 + recurse(m - 1, n)
            delCost = 1 + recurse(m, n - 1)
            result = min(subCost, insCost, delCost)
        cache[(m, n)] = result
        return result

    return recurse(len(s), len(t))


# print(computeEditDistance('a cat!', 'the cats!'))
print(computeEditDistance('a cat!' * 10, 'the cats!' * 10))

