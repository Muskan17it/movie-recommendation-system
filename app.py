import streamlit as st
import pandas as pd
import joblib
import urllib.parse


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_data():

    user_similarity_df = joblib.load("user_similarity.pkl")
    user_movie = joblib.load("user_movie.pkl")
    movies_data = pd.read_csv("movies_data.csv")

    return user_similarity_df, user_movie, movies_data


user_similarity_df, user_movie, movies_data = load_data()


# ============================================================
# MOVIE DETAILS
# ============================================================

movie_details = (
    movies_data
    .groupby("Movie_Key", as_index=False)
    .agg(
        Movie_Title=("Movie_Title", "first"),
        Release_Year=("Release_Year", "first"),
        Genre=("Genre", "first"),
        Language=("Language", "first"),
        Rating=("Rating", "mean")
    )
)


# ============================================================
# TITLE NORMALIZER
# ============================================================

def normalize_title(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# MOVIE IMAGE
# ============================================================

def get_movie_image(movie_title, release_year):

    if pd.isna(movie_title):
        return None

    movie_title = str(movie_title).strip()

    if movie_title == "":
        return None

    if pd.isna(release_year):

        search_text = (
            movie_title
            + " movie poster"
        )

    else:

        search_text = (
            movie_title
            + " "
            + str(int(release_year))
            + " movie poster"
        )

    query = urllib.parse.quote(search_text)

    return (
        "https://tse1.mm.bing.net/th"
        "?q="
        + query
        + "&w=300&h=450&c=7&rs=1"
    )


# ============================================================
# MODE 1
# USER-BASED COLLABORATIVE FILTERING
# ============================================================

def recommend_by_user(user_id, n_movies=5):

    n_movies = min(n_movies, 10)

    # --------------------------------------------------------
    # FIND SIMILAR USERS
    # --------------------------------------------------------

    similar_users = (
        user_similarity_df
        .loc[user_id]
        .sort_values(ascending=False)
        .iloc[1:]
    )

    similar_users = similar_users[
        similar_users > 0
    ]

    if similar_users.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # GET MOVIES OF SIMILAR USERS
    # --------------------------------------------------------

    candidate_movies = user_movie[
        user_movie["User_ID"].isin(
            similar_users.index
        )
    ].copy()

    if candidate_movies.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # ADD SIMILARITY
    # --------------------------------------------------------

    candidate_movies["Similarity"] = (
        candidate_movies["User_ID"]
        .map(similar_users)
    )

    # --------------------------------------------------------
    # REMOVE ALREADY WATCHED MOVIES
    # --------------------------------------------------------

    watched_movies = set(
        user_movie[
            user_movie["User_ID"] == user_id
        ]["Movie_Key"]
    )

    candidate_movies = candidate_movies[
        ~candidate_movies["Movie_Key"].isin(
            watched_movies
        )
    ]

    if candidate_movies.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # WEIGHTED RATING
    # --------------------------------------------------------

    candidate_movies["Weighted_Rating"] = (
        candidate_movies["Rating"]
        *
        candidate_movies["Similarity"]
    )

    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    recommendations = (
        candidate_movies
        .groupby("Movie_Key")
        .agg(
            Weighted_Rating=(
                "Weighted_Rating",
                "sum"
            ),
            Total_Similarity=(
                "Similarity",
                "sum"
            ),
            Number_of_Viewers=(
                "User_ID",
                "nunique"
            )
        )
        .reset_index()
    )

    recommendations = recommendations[
        recommendations["Total_Similarity"] > 0
    ]

    if recommendations.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    recommendations["Recommendation_Score"] = (
        recommendations["Weighted_Rating"]
        /
        recommendations["Total_Similarity"]
    )

    # --------------------------------------------------------
    # MOVIE DETAILS
    # --------------------------------------------------------

    recommendations = recommendations.merge(
        movie_details,
        on="Movie_Key",
        how="left"
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    recommendations = (
        recommendations
        .sort_values(
            [
                "Recommendation_Score",
                "Number_of_Viewers"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE TITLES
    # --------------------------------------------------------

    recommendations["Title_Check"] = (
        normalize_title(
            recommendations["Movie_Title"]
        )
    )

    recommendations = (
        recommendations
        .drop_duplicates(
            subset=["Title_Check"],
            keep="first"
        )
    )

    return (
        recommendations
        .head(n_movies)
        .reset_index(drop=True)
    )


# ============================================================
# MODE 2
# USER + RECENTLY WATCHED MOVIE
#
# MAIN LOGIC:
#
# User
#   ↓
# Recently watched movie
#   ↓
# Other users who watched same movie
#   ↓
# Their other movies
#   ↓
# Remove user's watched movies
#   ↓
# Rank
#   ↓
# If not enough → controlled expansion
# ============================================================

def recommend_from_recent_movie(
    user_id,
    movie_key,
    n_movies=5
):

    n_movies = min(n_movies, 10)

    # ========================================================
    # TARGET USER'S WATCHED MOVIES
    # ========================================================

    watched_movies = set(
        user_movie[
            user_movie["User_ID"] == user_id
        ]["Movie_Key"]
    )

    # ========================================================
    # GET SELECTED MOVIE TITLE
    # ========================================================

    selected_info = movie_details[
        movie_details["Movie_Key"] == movie_key
    ]

    if selected_info.empty:
        return pd.DataFrame()

    selected_title = (
        selected_info.iloc[0]["Movie_Title"]
    )

    if pd.isna(selected_title):
        return pd.DataFrame()

    selected_title = (
        str(selected_title)
        .strip()
        .lower()
    )

    # ========================================================
    # FIND ALL MOVIE KEYS WITH SAME TITLE
    # ========================================================

    title_map = (
        movie_details[
            [
                "Movie_Key",
                "Movie_Title"
            ]
        ]
        .dropna(
            subset=["Movie_Title"]
        )
        .copy()
    )

    title_map["Title_Check"] = (
        title_map["Movie_Title"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    same_title_keys = title_map[
        title_map["Title_Check"]
        == selected_title
    ]["Movie_Key"].tolist()

    # ========================================================
    # FIND OTHER USERS WHO WATCHED SAME MOVIE TITLE
    # ========================================================

    co_watchers = user_movie[
        (
            user_movie["Movie_Key"]
            .isin(same_title_keys)
        )
        &
        (
            user_movie["User_ID"]
            != user_id
        )
    ].copy()

    # ========================================================
    # CONTROLLED EXPANSION
    #
    # If very few people watched the movie,
    # use the strongest similar users as an additional
    # candidate source.
    #
    # This is ONLY used to fill missing recommendation slots.
    # ========================================================

    co_watcher_ids = set(
        co_watchers["User_ID"]
    )

    # ========================================================
    # GET USER SIMILARITIES
    # ========================================================

    similarity_series = (
        user_similarity_df
        .loc[user_id]
        .drop(index=user_id, errors="ignore")
        .sort_values(
            ascending=False
        )
    )

    # ========================================================
    # SOURCE 1 — DIRECT CO-WATCHERS
    # ========================================================

    direct_users = list(
        co_watcher_ids
    )

    # ========================================================
    # BUILD CANDIDATE MOVIES
    # ========================================================

    candidate_parts = []

    if direct_users:

        direct_movies = user_movie[
            user_movie["User_ID"].isin(
                direct_users
            )
        ].copy()

        if not direct_movies.empty:

            direct_movies["Source"] = (
                "co_watcher"
            )

            candidate_parts.append(
                direct_movies
            )

    # ========================================================
    # SOURCE 2 — SIMILAR USERS
    #
    # Only used when direct co-watchers don't provide
    # enough unique movies.
    # ========================================================

    if candidate_parts:

        temp_candidates = pd.concat(
            candidate_parts,
            ignore_index=True
        )

        temp_candidates = temp_candidates[
            ~temp_candidates["Movie_Key"]
            .isin(watched_movies)
        ]

        temp_candidates = temp_candidates[
            ~temp_candidates["Movie_Key"]
            .isin(same_title_keys)
        ]

        temp_candidates["Title_Check"] = (
            temp_candidates["Movie_Key"]
        )

        available_direct_titles = (
            temp_candidates["Movie_Key"]
            .nunique()
        )

    else:

        available_direct_titles = 0

    # Need more movies than direct co-watchers can provide
    if available_direct_titles < n_movies:

        # Take strongest similar users
        extra_users = []

        for candidate_user in similarity_series.index:

            if candidate_user in co_watcher_ids:
                continue

            similarity_value = (
                similarity_series[
                    candidate_user
                ]
            )

            if similarity_value <= 0:
                continue

            extra_users.append(
                candidate_user
            )

            # Don't need every user
            if len(extra_users) >= 30:
                break

        if extra_users:

            extra_movies = user_movie[
                user_movie["User_ID"].isin(
                    extra_users
                )
            ].copy()

            if not extra_movies.empty:

                extra_movies["Source"] = (
                    "similar_user"
                )

                candidate_parts.append(
                    extra_movies
                )

    # ========================================================
    # NO CANDIDATES
    # ========================================================

    if not candidate_parts:
        return pd.DataFrame()

    candidate_movies = pd.concat(
        candidate_parts,
        ignore_index=True
    )

    # ========================================================
    # REMOVE:
    #
    # 1. Movies already watched
    # 2. Selected movie
    # 3. Same movie title
    # ========================================================

    candidate_movies = candidate_movies[
        ~candidate_movies["Movie_Key"]
        .isin(watched_movies)
    ]

    candidate_movies = candidate_movies[
        ~candidate_movies["Movie_Key"]
        .isin(same_title_keys)
    ]

    if candidate_movies.empty:
        return pd.DataFrame()

    # ========================================================
    # ADD USER SIMILARITY
    # ========================================================

    candidate_movies["Similarity"] = (
        candidate_movies["User_ID"]
        .map(similarity_series)
        .fillna(0)
    )

    # ========================================================
    # GIVE DIRECT CO-WATCHERS EXTRA IMPORTANCE
    #
    # Someone who actually watched the selected movie
    # is more relevant than a generic similar user.
    # ========================================================

    candidate_movies["Base_Weight"] = (
        candidate_movies["Similarity"]
        + 0.1
    )

    candidate_movies.loc[
        candidate_movies["Source"]
        == "co_watcher",
        "Base_Weight"
    ] += 0.5

    # ========================================================
    # WEIGHTED RATING
    # ========================================================

    candidate_movies["Weighted_Rating"] = (
        candidate_movies["Rating"]
        *
        candidate_movies["Base_Weight"]
    )

    # ========================================================
    # AGGREGATE MOVIES
    # ========================================================

    recommendations = (
        candidate_movies
        .groupby("Movie_Key")
        .agg(
            Weighted_Rating=(
                "Weighted_Rating",
                "sum"
            ),
            Total_Weight=(
                "Base_Weight",
                "sum"
            ),
            Number_of_Viewers=(
                "User_ID",
                "nunique"
            ),
            Direct_CoWatchers=(
                "Source",
                lambda x:
                (x == "co_watcher").sum()
            ),
            Average_Rating=(
                "Rating",
                "mean"
            )
        )
        .reset_index()
    )

    # ========================================================
    # SCORE
    # ========================================================

    recommendations["Recommendation_Score"] = (
        recommendations["Weighted_Rating"]
        /
        recommendations["Total_Weight"]
    )

    # ========================================================
    # BOOST MOVIES SEEN BY MULTIPLE PEOPLE
    # ========================================================

    recommendations["Recommendation_Score"] += (
        recommendations[
            "Number_of_Viewers"
        ]
        * 0.08
    )

    recommendations["Recommendation_Score"] += (
        recommendations[
            "Direct_CoWatchers"
        ]
        * 0.20
    )

    # ========================================================
    # MOVIE DETAILS
    # ========================================================

    recommendations = recommendations.merge(
        movie_details,
        on="Movie_Key",
        how="left"
    )

    # ========================================================
    # SORT
    # ========================================================

    recommendations = (
        recommendations
        .sort_values(
            [
                "Recommendation_Score",
                "Direct_CoWatchers",
                "Number_of_Viewers",
                "Average_Rating"
            ],
            ascending=[
                False,
                False,
                False,
                False
            ]
        )
    )

    # ========================================================
    # REMOVE DUPLICATE MOVIE TITLES
    # ========================================================

    recommendations["Title_Check"] = (
        recommendations["Movie_Title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    recommendations = (
        recommendations
        .drop_duplicates(
            subset=["Title_Check"],
            keep="first"
        )
    )

    # ========================================================
    # RETURN REQUESTED NUMBER
    # ========================================================

    return (
        recommendations
        .head(n_movies)
        .reset_index(drop=True)
    )


# ============================================================
# GET USER'S MOVIES
# ============================================================

def get_user_movies(user_id):

    user_movies = user_movie[
        user_movie["User_ID"] == user_id
    ][
        ["Movie_Key"]
    ].drop_duplicates()

    user_movies = (
        user_movies
        .merge(
            movie_details,
            on="Movie_Key",
            how="left"
        )
    )

    user_movies = (
        user_movies
        .sort_values(
            "Movie_Title"
        )
    )

    return user_movies


# ============================================================
# GET MOVIE INFO
# ============================================================

def get_movie_info(movie_key):

    result = movie_details[
        movie_details["Movie_Key"]
        == movie_key
    ]

    if result.empty:
        return None

    return result.iloc[0]


# ============================================================
# FEATURED RECENTLY WATCHED MOVIE
# ============================================================

def render_featured_movie(
    movie_key,
    user_id
):

    movie = get_movie_info(
        movie_key
    )

    if movie is None:
        return

    movie_title = movie[
        "Movie_Title"
    ]

    if pd.isna(movie_title):

        movie_title = (
            movie_key
            .rsplit("_", 2)[0]
        )

    movie_title = str(
        movie_title
    )

    release_year = movie[
        "Release_Year"
    ]

    genre = movie[
        "Genre"
    ]

    language = movie[
        "Language"
    ]

    rating = movie[
        "Rating"
    ]

    if pd.isna(genre):
        genre = "Unknown"

    if pd.isna(language):
        language = "Unknown"

    image_url = get_movie_image(
        movie_title,
        release_year
    )

    st.subheader(
        "🎬 Recently Watched"
    )

    left, middle, right = st.columns(
        [1.1, 2.5, 1.8],
        gap="large"
    )

    with left:

        if image_url:

            st.image(
                image_url,
                use_container_width=True
            )

        else:

            st.info(
                "🎬 Poster unavailable"
            )

    with middle:

        st.subheader(
            movie_title
        )

        if not pd.isna(
            release_year
        ):

            st.caption(
                f"📅 {int(release_year)}"
            )

        st.write(
            f"🎭 **{genre}**"
        )

        st.write(
            f"🌐 **{language}**"
        )

        if not pd.isna(rating):

            st.metric(
                "⭐ Average Rating",
                f"{rating:.1f}/10"
            )

    with right:

        selected_rating = user_movie[
            (
                user_movie["User_ID"]
                == user_id
            )
            &
            (
                user_movie["Movie_Key"]
                == movie_key
            )
        ]

        if not selected_rating.empty:

            user_rating = (
                selected_rating.iloc[0][
                    "Rating"
                ]
            )

            st.success(
                f"👤 Viewer {user_id}"
            )

            st.metric(
                "Your Rating",
                f"{user_rating:.1f}/10"
            )

        co_watchers = user_movie[
            (
                user_movie["Movie_Key"]
                .isin(
                    movie_details[
                        normalize_title(
                            movie_details[
                                "Movie_Title"
                            ]
                        )
                        ==
                        str(movie_title)
                        .strip()
                        .lower()
                    ]["Movie_Key"]
                )
            )
            &
            (
                user_movie["User_ID"]
                != user_id
            )
        ]["User_ID"].nunique()

        st.caption(
            f"👥 {co_watchers} other viewer(s) "
            f"also watched this movie"
        )


# ============================================================
# MOVIE CARDS
# ============================================================

def render_movie_cards(
    recommendations
):

    if recommendations.empty:
        return

    cards_per_row = 5

    for start in range(
        0,
        len(recommendations),
        cards_per_row
    ):

        batch = recommendations.iloc[
            start:
            start + cards_per_row
        ]

        columns = st.columns(
            cards_per_row,
            gap="small"
        )

        for column, (
            index,
            movie
        ) in zip(
            columns,
            batch.iterrows()
        ):

            with column:

                movie_title = movie[
                    "Movie_Title"
                ]

                if pd.isna(
                    movie_title
                ):

                    movie_title = (
                        movie["Movie_Key"]
                        .rsplit("_", 2)[0]
                    )

                movie_title = str(
                    movie_title
                )

                release_year = movie[
                    "Release_Year"
                ]

                genre = movie[
                    "Genre"
                ]

                language = movie[
                    "Language"
                ]

                rating = movie[
                    "Rating"
                ]

                if pd.isna(genre):
                    genre = "Unknown"

                if pd.isna(language):
                    language = "Unknown"

                image_url = get_movie_image(
                    movie_title,
                    release_year
                )

                with st.container(
                    border=True
                ):

                    if image_url:

                        st.image(
                            image_url,
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "🎬 Poster unavailable"
                        )

                    st.caption(
                        f"#{index + 1}  •  ✨ Recommended"
                    )

                    st.write(
                        f"🎬 **{movie_title}**"
                    )

                    st.caption(
                        f"🎭 {genre}"
                    )

                    st.caption(
                        f"🌐 {language}"
                    )

                    if (
                        not pd.isna(
                            release_year
                        )
                        and
                        not pd.isna(
                            rating
                        )
                    ):

                        st.caption(
                            f"📅 {int(release_year)}"
                            f"  •  ⭐ {rating:.1f}/10"
                        )

                    elif not pd.isna(
                        release_year
                    ):

                        st.caption(
                            f"📅 {int(release_year)}"
                        )

                    elif not pd.isna(
                        rating
                    ):

                        st.caption(
                            f"⭐ {rating:.1f}/10"
                        )

                    if (
                        "Number_of_Viewers"
                        in movie.index
                    ):

                        viewers = int(
                            movie[
                                "Number_of_Viewers"
                            ]
                        )

                        st.caption(
                            f"👥 {viewers} viewer(s)"
                        )

                    elif (
                        "Direct_CoWatchers"
                        in movie.index
                    ):

                        viewers = int(
                            movie[
                                "Direct_CoWatchers"
                            ]
                        )

                        st.caption(
                            f"👥 {viewers} co-watcher(s)"
                        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🎬 CineMatch"
    )

    st.caption(
        "Personalized Movie Recommendation System"
    )

    st.divider()

    st.subheader(
        "🔍 Recommendation Mode"
    )

    mode = st.radio(
        "Choose how to get recommendations:",
        [
            "🙍 By User",
            "🎞️ By Recently Watched Movie"
        ],
        index=0
    )

    st.divider()

    # ========================================================
    # USER MODE
    # ========================================================

    if mode == "🙍 By User":

        st.subheader(
            "👤 Select Viewer"
        )

        user_id = st.selectbox(
            "Viewer ID",
            sorted(
                user_movie[
                    "User_ID"
                ].unique()
            ),
            label_visibility="collapsed"
        )

        rated_count = len(
            user_movie[
                user_movie["User_ID"]
                == user_id
            ]
        )

        st.caption(
            f"📊 {rated_count} movies rated"
        )

        selected_movie_key = None

    # ========================================================
    # RECENT MOVIE MODE
    # ========================================================

    else:

        st.subheader(
            "👤 Select Viewer"
        )

        user_id = st.selectbox(
            "Viewer ID",
            sorted(
                user_movie[
                    "User_ID"
                ].unique()
            ),
            label_visibility="collapsed",
            key="recent_user"
        )

        user_movies = get_user_movies(
            user_id
        )

        st.subheader(
            "🎬 What did you just watch?"
        )

        if user_movies.empty:

            st.warning(
                "This viewer has no movie history."
            )

            selected_movie_key = None

        else:

            movie_options = {}

            for _, row in user_movies.iterrows():

                title = row[
                    "Movie_Title"
                ]

                if pd.isna(title):
                    continue

                title = str(
                    title
                )

                year = row[
                    "Release_Year"
                ]

                if not pd.isna(year):

                    display_name = (
                        f"{title} ({int(year)})"
                    )

                else:

                    display_name = title

                movie_options[
                    display_name
                ] = row[
                    "Movie_Key"
                ]

            selected_label = st.selectbox(
                "Recently watched movie",
                sorted(
                    movie_options.keys()
                ),
                label_visibility="collapsed",
                key="recent_movie"
            )

            selected_movie_key = (
                movie_options[
                    selected_label
                ]
            )

            selected_rating = user_movie[
                (
                    user_movie["User_ID"]
                    == user_id
                )
                &
                (
                    user_movie["Movie_Key"]
                    == selected_movie_key
                )
            ]

            if not selected_rating.empty:

                rating_value = (
                    selected_rating.iloc[0][
                        "Rating"
                    ]
                )

                st.success(
                    f"⭐ Your rating: "
                    f"**{rating_value:.1f}/10**"
                )

    st.divider()

    number_of_movies = st.slider(
        "🎞️ Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()

    recommend = st.button(
        "✨ Get Recommendations",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🎬 CineMatch"
)

if mode == "🙍 By User":

    st.subheader(
        "Personalized Movie Recommendations"
    )

    st.write(
        "Discover movies based on viewers "
        "with similar preferences."
    )

else:

    st.subheader(
        "What Should You Watch Next?"
    )

    st.write(
        "Tell us what you just watched and "
        "discover what similar viewers watched next."
    )

st.divider()


# ============================================================
# RECOMMENDATIONS
# ============================================================

if recommend:

    # ========================================================
    # USER MODE
    # ========================================================

    if mode == "🙍 By User":

        with st.spinner(
            "🔎 Finding viewers with similar tastes..."
        ):

            recommendations = (
                recommend_by_user(
                    user_id,
                    number_of_movies
                )
            )

        if recommendations.empty:

            st.warning(
                f"No recommendations available "
                f"for Viewer {user_id}."
            )

        else:

            st.success(
                f"✨ Personalized picks for "
                f"Viewer {user_id}"
            )

            st.divider()

            st.subheader(
                "❤️ Movies You May Like"
            )

            st.caption(
                f"{len(recommendations)} recommendation(s)"
            )

            render_movie_cards(
                recommendations
            )

    # ========================================================
    # RECENT MOVIE MODE
    # ========================================================

    else:

        if selected_movie_key is None:

            st.warning(
                "Please select a recently watched movie."
            )

        else:

            with st.spinner(
                "🔎 Finding viewers with similar "
                "movie preferences..."
            ):

                recommendations = (
                    recommend_from_recent_movie(
                        user_id,
                        selected_movie_key,
                        number_of_movies
                    )
                )

            render_featured_movie(
                selected_movie_key,
                user_id
            )

            st.divider()

            if recommendations.empty:

                st.warning(
                    "No suitable recommendations "
                    "were found for this selection."
                )

            else:

                st.subheader(
                    "❤️ Movies You May Like"
                )

                st.caption(
                    "Based on other viewers who watched "
                    "the same movie and their other "
                    "movie preferences."
                )

                st.divider()

                render_movie_cards(
                    recommendations
                )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    if mode == "🙍 By User":

        st.info(
            "👈 Select a Viewer ID, choose the "
            "number of recommendations, then click "
            "**Get Recommendations**."
        )

    else:

        st.info(
            "👈 Select a Viewer, choose the movie "
            "you just finished watching, then click "
            "**Get Recommendations**."
        )