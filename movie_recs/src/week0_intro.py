import streamlit as st
import numpy as np
import pandas as pd

st.title('A Guide to an Insight Project with Streamlit')
st.write("""
Welcome to the Streamlit Guide for Insight fellows! In this guide, we walk you
through the lifecycle of an Insight ML project. You can think of this guide as
a "making-of" Amber Robert’s [Project Orient]
(https://github.com/AstronomerAmber/Project-Orient). Together, we will build a
movie recommendation system using the [MovieLens]
(https://grouplens.org/datasets/movielens/) dataset. Awesome, let's get started!

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
To unlock the rest of this guide, you'll need to check out the repo from
Github. See the instructions below for how to do that.
""")

st.header('Before We Start')
st.write("""
Before we start on this project, let's make sure a few things are set up first.

1. Do you have conda installed? If not, instructions are [here](https://www.anaconda.com/download/)
2. Do you have Atom installed? If not, download it from [here](https://atom.io/)
3. Do you have the Atom+Streamlit integration yet? If not, in Atom:
    1. Go to the Atom menu
    2. Select Preferences
    3. Select Install
    4. Type "streamlit"
    5. Click "Install" on the "streamlit-atom" package
""")

st.write("""
Though you can use Streamlit with any IDE, we **strongly**
recommend Atom. The remainder of the guide assumes you are using Atom with the
Streamlit package installed.
""")

st.header('How to Use This Guide')
st.write("""
To use the remainder of this guide,

0. Make sure you have conda, atom, and the atom+streamlit integration (see the
**Before We Start** section above).
1. Clone the streamlit-examples repo:
```git clone https://github.com/streamlit/streamlit-examples```
2. In Atom, open the streamlit-examples directory.
3. In your terminal, open the streamlit-examples directory, then change into:
```cd movies_recs```
4. In this directory, create the conda environment:
```conda env create -f movie_recs_env.yml```
5. Next, activate that environment with:
```conda activate movie_recs_env```
6. Change into the `src` directory:
```cd src```
7. Run `week1_explore.py`:
```python week1_explore.py```
7. In Atom, open week1_explore.py.
8. Open the Streamlit side pane (Ctrl+Alt+O). Your screen should
look something like this:
""")

from PIL import Image
image = Image.open("../static/week1-screenshot.png")
st.image(image, caption='Streamlit in Atom Screenshot', use_column_width=True)
