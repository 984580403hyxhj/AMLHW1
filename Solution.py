import math
from collections import defaultdict
from typing import List


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def pseudoPalindromicPaths(self, root) -> int:
        allNum = defaultdict(int)
        global result
        result = 0

        def check_pal(arr):
            num_odd = 0
            for k, v in arr:
                if arr[k] % 2 == 1:
                    num_odd += 1
            if num_odd > 1:
                return False
            return True

        def recurse(root):
            if root is None:
                if check_pal(allNum):
                    return 1
                return 0

            allNum[root.val] += 1

            recurse(root.left)
            recurse(root.right)
            allNum[root.val] -= 1
            return

        recurse(root)
        return result

a = {2:1, 3:2}
for i in a:
    print(i)

