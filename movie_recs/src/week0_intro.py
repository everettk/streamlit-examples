import streamlit as st
import numpy as np
import pandas as pd

st.title('Movie Recommendation with Streamlit')
st.write("""
Welcome to the Streamlit Guide for Insight fellows! Hopefully you reached this
point from [this document](https://docs.google.com/document/d/1XCsFFjrptPxMfr-HUU8QD154rGooemvHVZ95Q3iNZLc/edit?usp=sharing).
If not, make sure you read those instructions first.

In this guide, we walk you through the lifecycle of an Insight ML project. You
can think of this guide as a "making-of" Amber Robert’s [Project Orient](https://github.com/AstronomerAmber/Project-Orient). Together, we will build a
movie recommendation system using the [MovieLens]
(https://grouplens.org/datasets/movielens/) dataset. Awesome. Let's get started!

This project is broken into four parts, which correspond roughly to the weeks of
an Insight session.

1. **Exploring Data with Streamlit** - to better understand what we're working
with, we slice and dice and explore the MovieLens dataset.
2. **Recommendation System v0** - we build a naive recommendation system
end-to-end.
3. **Iterating on our recommendation system** - we experiment with two different
matrix factorization techniques to generate better recommendations.
4. **Running at scale** - we take our algorithm and run it on a dataset that is
200X larger. We run our report remotely on an AWS instance with a GPU.
""")

st.info("""
Next up: open week1_explore.py, and press ctrl+alt+r.
""")
