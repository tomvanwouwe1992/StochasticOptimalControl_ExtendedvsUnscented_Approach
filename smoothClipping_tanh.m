function x_clipped = smoothClipping_tanh(x,sharpnessFactor,lowerClippingValue,upperClippingValue)

x_clipped = x - (x-upperClippingValue).*(0.5*tanh(sharpnessFactor*(x-upperClippingValue)) + 0.5) - (x-lowerClippingValue).*(0.5*tanh(-sharpnessFactor*(x-lowerClippingValue)) + 0.5);


