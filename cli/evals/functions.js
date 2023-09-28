Sure! Here's a JavaScript function that lists the numbers above or equal to 3:

```javascript
function listNumbers(numbers) {
  var result = [];
  
  for (var i = 0; i < numbers.length; i++) {
    if (numbers[i] >= 3) {
      result.push(numbers[i]);
    }
  }
  
  return result;
}
```

In this function, the `listNumbers` function takes an array of numbers as a parameter. It initializes an empty array called `result` to store the numbers that are above or equal to 3. 

Then, it loops through each element in the `numbers` array using a `for` loop. If the current number is greater than or equal to 3, it is added to the `result` array using the `push` method.

Finally, the function returns the `result` array containing the numbers that are above or equal to 3.