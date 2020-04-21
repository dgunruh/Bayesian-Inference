import numpy as np
import matplotlib.pyplot as plt
import math

#######################
# Answer to Question 1#
#######################

# When we integrate over a^(-x^2/2) from
# -infinity to +infinity, we get the result
# sqrt(2π)/sqrt(ln(a))
# Thus, the normalization constant should be
# sqrt(ln(a))/sqrt(2π)

#######################
# Answer to Question 3#
#######################

# From Bayes theorem:
# P(a|{x}) = P({x}|a)P(a)/P({x})

# If we assume that P({x}) is our normalization
# constant, then we can ignore it and say that:
# P(a|{x}) = P({x}|a)P(a)

# Also note that P({x}|a) = Product{P(x|a)} from i=1 to n
# where n is the number of points in {x}.

# Finally we can assume P(a) is a uniform distribution,
# independent of a. This mean that it also becomes
# a normalization constant, which we can set equal to 1.

######################
# Answer to Question 4#
######################

# Now let's start by drawing our 100 samples
x = np.random.normal(0, 1.0, 100)
a = np.linspace(1.5, 5.0, 1000)  # 1000 linear values of a
P = np.zeros(1000)
for n, i in enumerate(a):
    p = 1
    for j in x:
        p = (
            p
            * math.sqrt(math.log(i))
            / math.sqrt(2 * math.pi)
            * (i ** (-(j ** 2) / 2.0))
        )
    P[n] = p * 10 ** 59

plt.plot(a, P)
plt.show()

######################
# Answer to Question 5#
######################

# Define the functions that we will need for our algorithm
# 1: P(a|{x})
def p_of_a(a, x):
    p = 1
    for j in x:
        p = (
            p
            * math.sqrt(math.log(a))
            / math.sqrt(2 * math.pi)
            * (a ** (-(j ** 2) / 2.0))
        )
    return p


# 2: our generating function, g(a1|a2)
def g(a1, a2, variance):
    return (
        1
        / math.sqrt(2 * math.pi * variance)
        * math.exp(-((a1 - a2) ** 2) / (2 * variance))
    )


# Define our Metropolis-Hastings algorithm input variables:
# starting parameter, and variance of the generating function
a0 = 4
variance = 1
aList = []

# Now we create our Markov chain. Run the MH algorithm
# for 2000 steps for each value n.
# Note: more steps may yield better convergence, this was
# not thoroughly tested (though 2000 steps was better than 1000)
for n in range(50, 500):
    x = np.random.normal(0, 1.0, n)
    for t in range(2000):
        new_a = np.random.normal(a0, math.sqrt(variance))
        while new_a < 1:
            new_a = np.random.normal(a0, math.sqrt(variance))
        weight = min(
            1,
            p_of_a(new_a, x)
            * g(a0, new_a, variance)
            / (p_of_a(a0, x) * g(new_a, a0, variance)),
        )
        r = np.random.uniform(0, 1)
        if r <= weight:
            # accept!
            a0 = new_a
    aList.append(a0)


######################
# Answer to Question 6#
######################
# Convert the outputted list of parameters into an array
# and plot n vs a (our trace plot)
aArray = np.asarray(aList)
plt.plot(range(50, 500), aArray)
plt.show()

######################
# Answer to Question 7#
######################

# Now we plot the histogram of all a values, and
# compare to P(a{x}), for len({x}) = 500
plt.hist(aArray, "fd")
x = np.random.normal(0, 1.0, 500)
a = np.linspace(1.5, 5.0, 1000)  # 1000 linear values of a
P = np.zeros(1000)
for n, i in enumerate(a):
    p = 1
    for j in x:
        p = (
            p
            * math.sqrt(math.log(i))
            / math.sqrt(2 * math.pi)
            * (i ** (-(j ** 2) / 2.0))
            * 4
        )
    P[n] = p
scale = 60 / np.amax(P)
P = np.multiply(P, scale)
plt.plot(a, P)
plt.show()

# We find that they do resemble each other, though P(a|{x}) varies from run to run
# and is in general skinnier (since we chose a large size)

