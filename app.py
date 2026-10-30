import streamlit as st
import pandas as pd
import joblib
import urllib.parse


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="MovieMind - AI Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
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

def get_user_fallback(user_id, watched_movies, needed=5):
    """Fallback when collaborative filtering finds 0 candidate movies."""
    user_watched_df = user_movie[user_movie["User_ID"] == user_id].merge(movie_details, on="Movie_Key", how="left")
    fav_genres = [g for g in user_watched_df["Genre"].dropna().value_counts().index if str(g).lower() != "unknown"]

    movie_stats = user_movie.groupby("Movie_Key").agg(
        Avg_Rating=("Rating", "mean"),
        Number_of_Viewers=("User_ID", "nunique")
    ).reset_index()

    candidates = movie_stats[~movie_stats["Movie_Key"].isin(watched_movies)].merge(movie_details, on="Movie_Key", how="left")
    candidates["Recommendation_Score"] = candidates["Avg_Rating"]
    candidates["Weighted_Rating"] = candidates["Avg_Rating"]
    candidates["Total_Similarity"] = 1.0

    if fav_genres:
        candidates["Genre_Match"] = candidates["Genre"].isin(fav_genres[:2]).astype(int)
        candidates = candidates.sort_values(by=["Genre_Match", "Recommendation_Score", "Number_of_Viewers"], ascending=[False, False, False])
    else:
        candidates = candidates.sort_values(by=["Recommendation_Score", "Number_of_Viewers"], ascending=[False, False])

    candidates["Title_Check"] = normalize_title(candidates["Movie_Title"])
    candidates = candidates.drop_duplicates(subset=["Title_Check"], keep="first")
    return candidates.head(needed).reset_index(drop=True)


def recommend_by_user(user_id, n_movies=5):

    n_movies = min(n_movies, 10)
    watched_movies = set(
        user_movie[
            user_movie["User_ID"] == user_id
        ]["Movie_Key"]
    )

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
        return get_user_fallback(user_id, watched_movies, n_movies)

    # --------------------------------------------------------
    # GET MOVIES OF SIMILAR USERS
    # --------------------------------------------------------

    candidate_movies = user_movie[
        user_movie["User_ID"].isin(
            similar_users.index
        )
    ].copy()

    if candidate_movies.empty:
        return get_user_fallback(user_id, watched_movies, n_movies)

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

    candidate_movies = candidate_movies[
        ~candidate_movies["Movie_Key"].isin(
            watched_movies
        )
    ]

    if candidate_movies.empty:
        return get_user_fallback(user_id, watched_movies, n_movies)

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
        return get_user_fallback(user_id, watched_movies, n_movies)

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

    # If CF produced fewer than n_movies, supplement with fallback
    if len(recommendations) < n_movies:
        existing_keys = set(recommendations["Movie_Key"]) | watched_movies
        fallback = get_user_fallback(user_id, existing_keys, n_movies - len(recommendations))
        if not fallback.empty:
            recommendations = pd.concat([recommendations, fallback], ignore_index=True)

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
# HELPER FUNCTIONS
# ============================================================

def render_stars(rating):
    if pd.isna(rating):
        return ""
    s     = rating / 2.0
    full  = int(s)
    half  = 1 if (s - full) >= 0.5 else 0
    empty = 5 - full - half
    return "\u2605" * full + ("\u00bd" if half else "") + "\u2606" * empty


def get_badge(index, rating, viewers):
    if index == 0:
        return "<span class='badge badge-gold'>\U0001f3c6 Top Pick</span>"
    if not pd.isna(rating) and rating >= 8.0:
        return "<span class='badge badge-green'>\u2b50 Top Rated</span>"
    if not pd.isna(viewers) and viewers >= 10:
        return "<span class='badge badge-purple'>\U0001f525 Trending</span>"
    if index < 3:
        return "<span class='badge badge-blue'>\u2728 Popular</span>"
    return "<span class='badge badge-gray'>\U0001f3ac Recommended</span>"


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700;800&display=swap');
:root{
    --bg-main:#050816;--bg-sidebar:#080D1C;--bg-card:#0C1426;
    --accent-primary:#8B5CF6;--accent-secondary:#D946EF;--accent-highlight:#A78BFA;
    --text-primary:#F8FAFC;--text-secondary:#94A3B8;--text-muted:#64748B;
    --rating-color:#FBBF24;--border:rgba(139,92,246,0.18);--border-hover:rgba(139,92,246,0.45);
    --glow-purple:rgba(139,92,246,0.25);--radius-card:16px;--radius-btn:12px;
    --transition:all 0.3s cubic-bezier(0.4,0,0.2,1);
}
html,body,[data-testid="stApp"]{background:var(--bg-main)!important;color:var(--text-primary)!important;font-family:'Inter',sans-serif!important;}
#MainMenu,footer,header{visibility:hidden!important;}
[data-testid="stDecoration"]{display:none!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg-main);}
::-webkit-scrollbar-thumb{background:var(--accent-primary);border-radius:99px;}
.main .block-container{padding:1.5rem 2rem 3rem 2rem!important;max-width:1600px!important;background:transparent!important;}
[data-testid="stSidebar"]{background:var(--bg-sidebar)!important;border-right:1px solid var(--border)!important;min-width:290px!important;}
[data-testid="stSidebar"]>div{background:transparent!important;}
[data-testid="stSidebarContent"]{padding:1.5rem 1.2rem!important;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span{color:var(--text-secondary)!important;font-family:'Inter',sans-serif!important;}
[data-testid="stSelectbox"]>div>div{background:rgba(12,20,38,0.8)!important;border:1px solid var(--border)!important;border-radius:var(--radius-btn)!important;color:var(--text-primary)!important;transition:var(--transition)!important;}
[data-testid="stSelectbox"]>div>div:hover{border-color:var(--accent-primary)!important;box-shadow:0 0 12px var(--glow-purple)!important;}
[data-testid="stRadio"] label{background:rgba(12,20,38,0.7)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:0.5rem 0.8rem!important;transition:var(--transition)!important;cursor:pointer!important;}
[data-testid="stRadio"] label:hover{border-color:var(--accent-primary)!important;background:rgba(139,92,246,0.1)!important;}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:1rem 0!important;}
[data-testid="stButton"]>button{background:linear-gradient(135deg,var(--accent-primary) 0%,var(--accent-secondary) 100%)!important;color:#fff!important;border:none!important;border-radius:var(--radius-btn)!important;font-family:'Outfit',sans-serif!important;font-weight:600!important;font-size:0.95rem!important;padding:0.65rem 1.5rem!important;transition:var(--transition)!important;box-shadow:0 4px 20px rgba(139,92,246,0.4)!important;}
[data-testid="stButton"]>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 30px rgba(139,92,246,0.6)!important;filter:brightness(1.1)!important;}
[data-testid="stAlert"]{background:rgba(12,20,38,0.8)!important;border:1px solid var(--border)!important;border-radius:var(--radius-card)!important;color:var(--text-primary)!important;}
[data-testid="stMetricValue"]{color:var(--accent-highlight)!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:var(--text-secondary)!important;}
[data-testid="stImage"] img{border-radius:12px!important;object-fit:cover!important;}
[data-testid="stCaptionContainer"] p{color:var(--text-muted)!important;font-size:0.78rem!important;}

/* TOP NAV */
.top-nav{display:flex;align-items:center;justify-content:space-between;background:rgba(8,13,28,0.85);backdrop-filter:blur(20px);border:1px solid var(--border);border-radius:16px;padding:0.75rem 1.5rem;margin-bottom:1.5rem;box-shadow:0 4px 30px rgba(0,0,0,0.4);}
.top-nav-brand{font-family:'Outfit',sans-serif;font-size:1.35rem;font-weight:800;background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.top-nav-right{display:flex;align-items:center;gap:1rem;}
.nav-avatar{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary));display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;color:white;box-shadow:0 0 12px var(--glow-purple);}
/* SIDEBAR */
.sidebar-logo{display:flex;align-items:center;gap:0.75rem;margin-bottom:0.25rem;}
.sidebar-logo-icon{width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary));display:flex;align-items:center;justify-content:center;font-size:1.3rem;box-shadow:0 4px 15px var(--glow-purple);flex-shrink:0;}
.sidebar-logo-text h2{margin:0;padding:0;font-family:'Outfit',sans-serif;font-size:1.3rem;font-weight:800;background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.sidebar-logo-text p{margin:0;padding:0;font-size:0.72rem;color:var(--text-muted)!important;font-style:italic;}
.sidebar-nav-item{display:flex;align-items:center;gap:0.75rem;padding:0.6rem 0.85rem;border-radius:10px;color:var(--text-secondary)!important;font-size:0.875rem;font-weight:500;cursor:pointer;transition:var(--transition);margin-bottom:0.2rem;border:1px solid transparent;}
.sidebar-nav-item:hover{background:rgba(139,92,246,0.12);color:var(--text-primary)!important;border-color:var(--border);}
.sidebar-nav-item.active{background:rgba(139,92,246,0.18);color:var(--accent-highlight)!important;border-color:rgba(139,92,246,0.3);}
.nav-count{margin-left:auto;background:rgba(139,92,246,0.2);color:var(--accent-highlight);border-radius:99px;padding:0.1rem 0.5rem;font-size:0.65rem;font-weight:700;}
.sidebar-section-title{font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted)!important;margin:1rem 0 0.5rem 0.3rem;}
.viewer-info-card{background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);border-radius:12px;padding:0.75rem 1rem;margin-top:0.5rem;display:flex;align-items:center;gap:0.75rem;}
.viewer-avatar{width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,#7C3AED,#DB2777);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.9rem;color:white;flex-shrink:0;}
.viewer-name{font-size:0.875rem;font-weight:600;color:var(--text-primary);}
.viewer-stats{font-size:0.72rem;color:var(--text-muted);margin-top:0.15rem;}
/* HERO */
.hero-section{position:relative;border-radius:20px;overflow:hidden;min-height:380px;background:var(--bg-card);border:1px solid var(--border);margin-bottom:2rem;box-shadow:0 8px 40px rgba(0,0,0,0.5);}
.hero-backdrop{position:absolute;inset:0;background:linear-gradient(135deg,#0C0B1A 0%,#1A0A2E 40%,#0D1B3E 70%,#050816 100%);z-index:0;}
.hero-overlay{position:absolute;inset:0;background:linear-gradient(90deg,rgba(5,8,22,0.97) 0%,rgba(5,8,22,0.75) 55%,rgba(5,8,22,0.3) 100%);z-index:1;}
.hero-content{position:relative;z-index:2;padding:2.5rem 3rem;display:flex;align-items:center;gap:2.5rem;min-height:380px;}
.hero-poster{flex-shrink:0;width:200px;height:300px;border-radius:14px;overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,0.6),0 0 0 1px var(--border);}
.hero-poster img{width:100%;height:100%;object-fit:cover;}
.hero-info{flex:1;}
.hero-badge{display:inline-flex;align-items:center;gap:0.35rem;background:rgba(139,92,246,0.2);border:1px solid rgba(139,92,246,0.4);border-radius:99px;padding:0.25rem 0.85rem;font-size:0.72rem;font-weight:600;color:var(--accent-highlight);letter-spacing:0.05em;text-transform:uppercase;margin-bottom:0.85rem;}
.hero-title{font-family:'Outfit',sans-serif;font-size:2.4rem;font-weight:800;color:var(--text-primary);line-height:1.15;margin-bottom:0.6rem;letter-spacing:-0.03em;}
.hero-meta{display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.85rem;}
.hero-meta-item{display:flex;align-items:center;gap:0.3rem;font-size:0.83rem;color:var(--text-secondary);background:rgba(255,255,255,0.05);border:1px solid var(--border);border-radius:99px;padding:0.25rem 0.65rem;}
.hero-stars{color:var(--rating-color);font-size:1rem;letter-spacing:0.05em;}
.hero-rating-val{font-size:0.85rem;font-weight:700;color:var(--rating-color);}
.hero-desc{font-size:0.875rem;color:var(--text-secondary);line-height:1.7;max-width:520px;margin-bottom:1.5rem;}
.hero-actions{display:flex;gap:0.85rem;flex-wrap:wrap;}
.hero-btn-primary{background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary));color:white;border:none;border-radius:99px;padding:0.6rem 1.5rem;font-size:0.875rem;font-weight:600;cursor:pointer;transition:var(--transition);display:inline-flex;align-items:center;gap:0.4rem;text-decoration:none;box-shadow:0 4px 20px rgba(139,92,246,0.4);font-family:'Inter',sans-serif;}
.hero-btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(139,92,246,0.6);color:white;text-decoration:none;}
.hero-btn-secondary{background:rgba(255,255,255,0.06);color:var(--text-primary);border:1px solid var(--border);border-radius:99px;padding:0.6rem 1.5rem;font-size:0.875rem;font-weight:500;cursor:pointer;transition:var(--transition);display:inline-flex;align-items:center;gap:0.4rem;text-decoration:none;font-family:'Inter',sans-serif;backdrop-filter:blur(10px);}
.hero-btn-secondary:hover{background:rgba(255,255,255,0.12);border-color:var(--accent-primary);color:var(--text-primary);text-decoration:none;transform:translateY(-2px);}
.hero-orb{position:absolute;border-radius:50%;filter:blur(60px);}
.hero-orb-1{width:300px;height:300px;background:rgba(139,92,246,0.15);top:-80px;right:5%;z-index:1;pointer-events:none;}
.hero-orb-2{width:200px;height:200px;background:rgba(217,70,239,0.1);bottom:-50px;right:25%;z-index:1;pointer-events:none;}
/* SECTION HEADER */
.section-title{font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text-primary);margin:0;letter-spacing:-0.02em;}
.section-subtitle{font-size:0.82rem;color:var(--text-muted);margin-top:0.25rem;}
.view-all-btn{font-size:0.8rem;font-weight:600;color:var(--accent-highlight);background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.3);border-radius:99px;padding:0.35rem 1rem;white-space:nowrap;}
/* MOVIE CARD */
.movie-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-card);overflow:hidden;transition:var(--transition);position:relative;cursor:pointer;height:100%;display:flex;flex-direction:column;}
.movie-card:hover{transform:translateY(-6px);border-color:var(--border-hover);box-shadow:0 12px 40px rgba(0,0,0,0.4),0 0 20px var(--glow-purple);}
.movie-card-poster{position:relative;overflow:hidden;aspect-ratio:2/3;background:linear-gradient(135deg,#1a1040,#0d1b3e);flex-shrink:0;}
.movie-card-poster img{width:100%;height:100%;object-fit:cover;transition:transform 0.4s ease;display:block;}
.movie-card:hover .movie-card-poster img{transform:scale(1.08);}
.movie-card-rank{position:absolute;top:0.5rem;left:0.5rem;background:rgba(5,8,22,0.85);backdrop-filter:blur(8px);border:1px solid var(--border);border-radius:99px;padding:0.2rem 0.6rem;font-size:0.7rem;font-weight:700;color:var(--accent-highlight);}
.movie-card-overlay{position:absolute;inset:0;background:linear-gradient(to top,rgba(5,8,22,0.95) 0%,rgba(5,8,22,0.4) 45%,transparent 70%);opacity:0;transition:opacity 0.35s ease;display:flex;align-items:flex-end;padding:1rem;}
.movie-card:hover .movie-card-overlay{opacity:1;}
.movie-card-trailer-btn{width:100%;background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary));color:white;border:none;border-radius:8px;padding:0.5rem;font-size:0.78rem;font-weight:600;cursor:pointer;text-align:center;text-decoration:none;font-family:'Inter',sans-serif;}
.movie-card-trailer-btn:hover{color:white;text-decoration:none;}
.movie-card-body{padding:0.85rem;flex:1;display:flex;flex-direction:column;gap:0.3rem;}
.movie-card-title{font-family:'Outfit',sans-serif;font-size:0.92rem;font-weight:700;color:var(--text-primary);line-height:1.3;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;}
.movie-card-meta{display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;}
.meta-pill{font-size:0.68rem;color:var(--text-muted);background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);border-radius:99px;padding:0.15rem 0.55rem;font-weight:500;}
.movie-card-rating{display:flex;align-items:center;gap:0.4rem;margin-top:0.1rem;}
.star-rating{color:var(--rating-color);font-size:0.78rem;letter-spacing:0.03em;}
.rating-val{font-size:0.75rem;font-weight:700;color:var(--rating-color);}
.movie-card-viewers{font-size:0.7rem;color:var(--text-muted);margin-top:auto;padding-top:0.4rem;}
/* BADGES */
.badge{display:inline-flex;align-items:center;gap:0.25rem;border-radius:99px;padding:0.18rem 0.55rem;font-size:0.65rem;font-weight:700;letter-spacing:0.03em;}
.badge-gold{background:rgba(251,191,36,0.15);color:#FBBF24;border:1px solid rgba(251,191,36,0.3);}
.badge-green{background:rgba(16,185,129,0.12);color:#34D399;border:1px solid rgba(16,185,129,0.25);}
.badge-purple{background:rgba(139,92,246,0.15);color:var(--accent-highlight);border:1px solid rgba(139,92,246,0.3);}
.badge-blue{background:rgba(59,130,246,0.12);color:#60A5FA;border:1px solid rgba(59,130,246,0.25);}
.badge-gray{background:rgba(100,116,139,0.12);color:var(--text-muted);border:1px solid rgba(100,116,139,0.2);}
/* FEATURED */
.featured-card{background:var(--bg-card);border:1px solid var(--border);border-radius:20px;overflow:hidden;margin-bottom:2rem;display:flex;gap:0;box-shadow:0 8px 40px rgba(0,0,0,0.4);}
.featured-poster{width:180px;flex-shrink:0;position:relative;overflow:hidden;}
.featured-poster img{width:100%;height:100%;object-fit:cover;display:block;}
.featured-body{padding:1.75rem 2rem;flex:1;display:flex;flex-direction:column;gap:0.6rem;}
.featured-label{font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:var(--accent-highlight);}
.featured-title{font-family:'Outfit',sans-serif;font-size:1.75rem;font-weight:800;color:var(--text-primary);letter-spacing:-0.02em;line-height:1.2;}
.featured-stat{background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:10px;padding:0.5rem 1rem;text-align:center;}
.featured-stat-val{font-family:'Outfit',sans-serif;font-size:1.2rem;font-weight:700;color:var(--accent-highlight);}
.featured-stat-label{font-size:0.68rem;color:var(--text-muted);margin-top:0.1rem;}
/* EMPTY STATE */
.empty-state{text-align:center;padding:4rem 2rem;background:var(--bg-card);border:1px solid var(--border);border-radius:20px;}
.empty-state-icon{font-size:3.5rem;margin-bottom:1rem;}
.empty-state-title{font-family:'Outfit',sans-serif;font-size:1.3rem;font-weight:700;color:var(--text-primary);margin-bottom:0.5rem;}
.empty-state-desc{font-size:0.875rem;color:var(--text-muted);max-width:380px;margin:0 auto;line-height:1.6;}
/* WELCOME */
.welcome-hero{text-align:center;padding:5rem 2rem;background:radial-gradient(ellipse at center top,rgba(139,92,246,0.12) 0%,transparent 70%),var(--bg-card);border:1px solid var(--border);border-radius:24px;position:relative;overflow:hidden;margin-top:1rem;}
.welcome-title{font-family:'Outfit',sans-serif;font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,var(--text-primary),var(--accent-highlight));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:0.75rem;letter-spacing:-0.03em;}
.welcome-desc{font-size:1rem;color:var(--text-secondary);max-width:480px;margin:0 auto 2rem auto;line-height:1.7;}
.welcome-step{background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:14px;padding:1.25rem 1.5rem;text-align:left;max-width:200px;transition:var(--transition);}
.welcome-step:hover{border-color:var(--accent-primary);background:rgba(139,92,246,0.08);transform:translateY(-3px);}
.welcome-step-num{font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:800;background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:0.4rem;}
.welcome-step-title{font-size:0.875rem;font-weight:600;color:var(--text-primary);margin-bottom:0.25rem;}
.welcome-step-desc{font-size:0.75rem;color:var(--text-muted);line-height:1.5;}
/* STATS */
.stats-row{display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap;}
.stat-card{flex:1;min-width:140px;background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:1rem 1.25rem;display:flex;align-items:center;gap:0.85rem;transition:var(--transition);}
.stat-card:hover{border-color:var(--accent-primary);box-shadow:0 4px 20px var(--glow-purple);transform:translateY(-2px);}
.stat-icon{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0;}
.stat-icon-purple{background:rgba(139,92,246,0.15);}
.stat-icon-pink{background:rgba(217,70,239,0.12);}
.stat-icon-gold{background:rgba(251,191,36,0.12);}
.stat-icon-teal{background:rgba(20,184,166,0.12);}
.stat-val{font-family:'Outfit',sans-serif;font-size:1.3rem;font-weight:700;color:var(--text-primary);}
.stat-label{font-size:0.72rem;color:var(--text-muted);margin-top:0.1rem;}
/* ANIMATIONS */
@keyframes fadeInUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
.fade-in-up{animation:fadeInUp 0.5s ease forwards;}
/* RESPONSIVE */
@media(max-width:768px){
    .hero-content{padding:1.5rem;flex-direction:column;}
    .hero-poster{width:140px;height:210px;}
    .hero-title{font-size:1.6rem;}
    .featured-card{flex-direction:column;}
    .featured-poster{width:100%;height:200px;}
    .welcome-title{font-size:1.6rem;}
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "wishlist" not in st.session_state:
    st.session_state.wishlist = set()
if "recommendations" not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()
if "last_mode" not in st.session_state:
    st.session_state.last_mode = None
if "hero_index" not in st.session_state:
    st.session_state.hero_index = 0
if "selected_movie_key" not in st.session_state:
    st.session_state.selected_movie_key = None
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">\U0001f3ac</div>
        <div class="sidebar-logo-text">
            <h2>MovieMind</h2>
            <p>Find Your Next Favorite</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    total_movies = movie_details.shape[0]
    total_users  = user_movie["User_ID"].nunique()

    # ── Functional nav buttons ────────────────────────────
    NAV_ITEMS = [
        ("\U0001f3e0", "Home",            ""),
        ("\u2728",      "Recommendations", "AI"),
        ("\U0001f525", "Trending",         ""),
        ("\u2764\ufe0f", "My Watchlist",   str(len(st.session_state.wishlist))),
        ("\U0001f550", "History",          ""),
    ]

    st.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 0.55rem 0.85rem !important;
        margin-bottom: 0.15rem !important;
        width: 100% !important;
        justify-content: flex-start !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        background: rgba(139,92,246,0.12) !important;
        border-color: rgba(139,92,246,0.3) !important;
        color: var(--text-primary) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    .nav-btn-active > button {
        background: rgba(139,92,246,0.18) !important;
        border-color: rgba(139,92,246,0.35) !important;
        color: #A78BFA !important;
    }
    </style>
    """, unsafe_allow_html=True)

    for icon, label, count in NAV_ITEMS:
        is_active = st.session_state.active_page == label
        badge     = f' \u00a0 [{count}]' if count else ''
        btn_label = f"{icon}  {label}{badge}"
        if is_active:
            st.markdown('<div class="nav-btn-active">', unsafe_allow_html=True)
        if st.button(btn_label, key=f"nav_{label}", use_container_width=True):
            st.session_state.active_page = label
            st.rerun()
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-section-title">Recommendation Mode</div>', unsafe_allow_html=True)

    mode = st.radio(
        "mode_selector",
        ["\U0001f9cd By User", "\U0001f39e\ufe0f By Recently Watched Movie"],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    if mode == "\U0001f9cd By User":

        st.markdown('<div class="sidebar-section-title">Select Viewer</div>', unsafe_allow_html=True)

        user_id = st.selectbox(
            "Viewer ID",
            sorted(user_movie["User_ID"].unique()),
            label_visibility="collapsed"
        )

        rated_count = len(user_movie[user_movie["User_ID"] == user_id])

        uid_display = str(user_id)[-2:] if len(str(user_id)) >= 2 else str(user_id)
        st.markdown(f"""
        <div class="viewer-info-card">
            <div class="viewer-avatar">V{uid_display}</div>
            <div>
                <div class="viewer-name">Viewer #{user_id}</div>
                <div class="viewer-stats">\U0001f4ca {rated_count} movies rated</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        selected_movie_key = None

    else:

        st.markdown('<div class="sidebar-section-title">Select Viewer</div>', unsafe_allow_html=True)

        user_id = st.selectbox(
            "Viewer ID",
            sorted(user_movie["User_ID"].unique()),
            label_visibility="collapsed",
            key="recent_user"
        )

        rated_count = len(user_movie[user_movie["User_ID"] == user_id])
        uid_display = str(user_id)[-2:] if len(str(user_id)) >= 2 else str(user_id)

        st.markdown(f"""
        <div class="viewer-info-card">
            <div class="viewer-avatar">V{uid_display}</div>
            <div>
                <div class="viewer-name">Viewer #{user_id}</div>
                <div class="viewer-stats">\U0001f4ca {rated_count} movies rated</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        user_movies_list = get_user_movies(user_id)

        st.markdown('<div class="sidebar-section-title" style="margin-top:1rem;">Recently Watched</div>', unsafe_allow_html=True)

        if user_movies_list.empty:
            st.warning("This viewer has no movie history.")
            selected_movie_key = None
        else:
            movie_options = {}
            for _, row in user_movies_list.iterrows():
                title = row["Movie_Title"]
                if pd.isna(title):
                    continue
                title = str(title)
                year  = row["Release_Year"]
                display_name = f"{title} ({int(year)})" if not pd.isna(year) else title
                movie_options[display_name] = row["Movie_Key"]

            selected_label = st.selectbox(
                "Recently watched movie",
                sorted(movie_options.keys()),
                label_visibility="collapsed",
                key="recent_movie"
            )
            selected_movie_key = movie_options[selected_label]

            sel_rating = user_movie[
                (user_movie["User_ID"] == user_id) &
                (user_movie["Movie_Key"] == selected_movie_key)
            ]
            if not sel_rating.empty:
                rv = sel_rating.iloc[0]["Rating"]
                st.markdown(f"""
                <div style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);border-radius:10px;padding:0.6rem 0.85rem;margin-top:0.5rem;">
                    <div style="font-size:0.7rem;color:var(--text-muted);margin-bottom:0.2rem;">Your Rating</div>
                    <div style="color:var(--rating-color);font-size:0.95rem;">{render_stars(rv)}
                        <span style="color:var(--text-primary);font-weight:700;font-size:0.85rem;"> {rv:.1f}/10</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-section-title">Number of Recommendations</div>', unsafe_allow_html=True)

    number_of_movies = st.slider(
        "n_recs",
        min_value=1,
        max_value=10,
        value=5,
        label_visibility="collapsed"
    )

    st.markdown(f"""
    <div style="text-align:center;font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;
                margin:-0.5rem 0 0.75rem 0;background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary));
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{number_of_movies}</div>
    """, unsafe_allow_html=True)

    st.divider()

    recommend = st.button("\u2728 Get Recommendations", use_container_width=True, type="primary")

    st.markdown(f"""
    <div style="margin-top:2rem;padding:0.85rem;background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:12px;text-align:center;">
        <div style="font-size:0.68rem;color:var(--text-muted);margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">Platform Stats</div>
        <div style="display:flex;justify-content:space-around;">
            <div><div style="font-family:'Outfit',sans-serif;font-size:1rem;font-weight:700;color:var(--accent-highlight);">{total_movies}</div><div style="font-size:0.65rem;color:var(--text-muted);">Movies</div></div>
            <div style="width:1px;background:var(--border);"></div>
            <div><div style="font-family:'Outfit',sans-serif;font-size:1rem;font-weight:700;color:var(--accent-highlight);">{total_users}</div><div style="font-size:0.65rem;color:var(--text-muted);">Viewers</div></div>
            <div style="width:1px;background:var(--border);"></div>
            <div><div style="font-family:'Outfit',sans-serif;font-size:1rem;font-weight:700;color:var(--accent-highlight);">AI</div><div style="font-size:0.65rem;color:var(--text-muted);">Powered</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TOP NAV BAR
# ============================================================

uid_display_nav = str(user_id)[-2:] if len(str(user_id)) >= 2 else str(user_id)
st.markdown(f"""
<div class="top-nav">
    <div class="top-nav-brand">\U0001f3ac MovieMind</div>
    <div style="flex:1;max-width:420px;margin:0 2rem;display:flex;align-items:center;
                background:rgba(12,20,38,0.7);border:1px solid var(--border);
                border-radius:99px;padding:0.45rem 1rem;gap:0.5rem;">
        <span style="color:var(--text-muted);font-size:0.9rem;">\U0001f50d</span>
        <span style="color:var(--text-muted);font-size:0.83rem;">Search for movies, genres, actors...</span>
    </div>
    <div class="top-nav-right">
        <div style="width:36px;height:36px;border-radius:50%;background:rgba(139,92,246,0.12);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:1rem;">\U0001f514</div>
        <div style="width:36px;height:36px;border-radius:50%;background:rgba(139,92,246,0.12);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:1rem;">\u2699\ufe0f</div>
        <div style="display:flex;align-items:center;gap:0.6rem;">
            <div class="nav-avatar">V{uid_display_nav}</div>
            <div>
                <div style="font-size:0.8rem;font-weight:600;color:var(--text-primary);">Viewer #{user_id}</div>
                <div style="font-size:0.68rem;color:var(--text-muted);">Movie Enthusiast</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# RECOMMENDATION LOGIC
# ============================================================

# Auto-prepare recommendations on first launch or when viewer/mode changes, or button clicked
user_changed = ("prev_user_id" in st.session_state and st.session_state.prev_user_id != user_id)
mode_changed = ("prev_mode" in st.session_state and st.session_state.prev_mode != mode)
needs_init   = st.session_state.recommendations.empty

if recommend or user_changed or mode_changed or needs_init:
    st.session_state.hero_index = 0
    if mode == "🧍 By User":
        with st.spinner("🔎 Analysing your preferences with AI..."):
            recs = recommend_by_user(user_id, number_of_movies)
        st.session_state.recommendations = recs
        st.session_state.last_mode = "user"
    else:
        if selected_movie_key is None:
            if recommend:
                st.warning("Please select a recently watched movie first.")
            st.session_state.recommendations = pd.DataFrame()
        else:
            with st.spinner("🔎 Finding viewers with similar tastes..."):
                recs = recommend_from_recent_movie(user_id, selected_movie_key, number_of_movies)
            st.session_state.recommendations = recs
            st.session_state.last_mode = "recent"
            st.session_state.selected_movie_key = selected_movie_key

st.session_state.prev_user_id = user_id
st.session_state.prev_mode = mode
recommendations = st.session_state.recommendations


# ============================================================
# HERO SECTION
# ============================================================

def render_hero(recommendations, user_id):
    if not recommendations.empty:
        hero_movies = recommendations.head(min(5, len(recommendations)))
        idx      = st.session_state.hero_index % len(hero_movies)
        hero_row = hero_movies.iloc[idx]
        title    = str(hero_row.get("Movie_Title", "")) if not pd.isna(hero_row.get("Movie_Title", None)) else "Featured Movie"
        year     = hero_row.get("Release_Year", None)
        genre    = str(hero_row.get("Genre", "Unknown")) if not pd.isna(hero_row.get("Genre", None)) else "Unknown"
        language = str(hero_row.get("Language", "Unknown")) if not pd.isna(hero_row.get("Language", None)) else "Unknown"
        rating   = hero_row.get("Rating", None)
        img_url  = get_movie_image(title, year)
        stars    = render_stars(rating) if rating is not None and not pd.isna(rating) else ""
        rating_v = f"{rating:.1f}/10" if rating is not None and not pd.isna(rating) else "N/A"
        year_str = str(int(year)) if year is not None and not pd.isna(year) else "N/A"
        score    = hero_row.get("Recommendation_Score", None)
        desc     = f"An AI-curated recommendation for Viewer #{user_id}. This {genre} film from {year_str} has been praised by viewers with similar taste profiles."
        n_heroes = len(hero_movies)
    else:
        title = "MovieMind"; genre = "All Genres"; language = "Multi-Language"
        rating = 9.0; img_url = None; stars = "\u2605\u2605\u2605\u2605\u2605"
        rating_v = "9.0/10"; year_str = "2024"; n_heroes = 1; idx = 0; score = None
        desc = "Your AI-powered cinematic companion. Discover personalized movie recommendations crafted uniquely for you."

    poster_html = (
        f'<img src="{img_url}" style="width:100%;height:100%;object-fit:cover;" />'
        if img_url
        else '<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:4rem;color:rgba(139,92,246,0.3);background:linear-gradient(135deg,#1a1040,#0d1b3e);">\U0001f3ac</div>'
    )

    dots_html = ""
    if n_heroes > 1:
        for i in range(n_heroes):
            sty = "background:var(--accent-primary);width:20px;" if i == idx else "background:rgba(255,255,255,0.25);width:8px;"
            dots_html += f'<div style="height:8px;border-radius:99px;transition:all 0.3s ease;{sty}"></div>'

    yt_query    = urllib.parse.quote(f"{title} official trailer")
    trailer_url = f"https://www.youtube.com/results?search_query={yt_query}"
    score_badge = (
        f'<span style="margin-left:0.5rem;background:rgba(139,92,246,0.2);border:1px solid rgba(139,92,246,0.4);border-radius:99px;padding:0.15rem 0.55rem;font-size:0.68rem;font-weight:700;color:var(--accent-highlight);">Score: {score:.2f}</span>'
        if score is not None else ""
    )

    st.markdown(f"""
    <div class="hero-section fade-in-up">
        <div class="hero-backdrop"></div>
        <div class="hero-orb hero-orb-1"></div>
        <div class="hero-orb hero-orb-2"></div>
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-poster">{poster_html}</div>
            <div class="hero-info">
                <div class="hero-badge">\u2728 AI Powered Recommendation</div>
                <div class="hero-title">{title}</div>
                <div class="hero-meta">
                    <div class="hero-meta-item"><span class="hero-stars">{stars}</span><span class="hero-rating-val">{rating_v}</span></div>
                    <div class="hero-meta-item">\U0001f4c5 {year_str}</div>
                    <div class="hero-meta-item">\U0001f3ad {genre}</div>
                    <div class="hero-meta-item">\U0001f310 {language}</div>
                    {score_badge}
                </div>
                <div class="hero-desc">{desc}</div>
                <div class="hero-actions">
                    <a href="{trailer_url}" target="_blank" class="hero-btn-primary">\u25b6 Watch Trailer</a>
                    <span class="hero-btn-secondary">\u2139 More Details</span>
                </div>
            </div>
        </div>
        <div style="position:absolute;bottom:1.25rem;right:1.5rem;z-index:3;display:flex;gap:0.4rem;align-items:center;">{dots_html}</div>
    </div>
    """, unsafe_allow_html=True)

    if n_heroes > 1:
        c1, c2, c3 = st.columns([1, 8, 1])
        with c1:
            if st.button("\u25c4", key="hero_prev"):
                st.session_state.hero_index = (st.session_state.hero_index - 1) % n_heroes
                st.rerun()
        with c3:
            if st.button("\u25ba", key="hero_next"):
                st.session_state.hero_index = (st.session_state.hero_index + 1) % n_heroes
                st.rerun()

# ============================================================
# RECENTLY WATCHED FEATURED CARD
# ============================================================

def render_featured_recent(user_id, sel_key, recommendations):
    movie = get_movie_info(sel_key)
    if movie is not None:
        m_title  = str(movie["Movie_Title"]) if not pd.isna(movie["Movie_Title"]) else sel_key.rsplit("_", 2)[0]
        m_year   = movie["Release_Year"]
        m_genre  = str(movie["Genre"]) if not pd.isna(movie["Genre"]) else "Unknown"
        m_lang   = str(movie["Language"]) if not pd.isna(movie["Language"]) else "Unknown"
        m_rating = movie["Rating"]
        m_img    = get_movie_image(m_title, m_year)
        m_stars  = render_stars(m_rating) if not pd.isna(m_rating) else ""
        m_year_s = str(int(m_year)) if not pd.isna(m_year) else "N/A"
        m_rat_s  = f"{m_rating:.1f}/10" if not pd.isna(m_rating) else "N/A"

        ur = user_movie[
            (user_movie["User_ID"] == user_id) &
            (user_movie["Movie_Key"] == sel_key)
        ]
        user_rating_val = ur.iloc[0]["Rating"] if not ur.empty else None

        co_w = user_movie[
            (user_movie["Movie_Key"].isin(
                movie_details[normalize_title(movie_details["Movie_Title"]) == m_title.strip().lower()]["Movie_Key"]
            )) & (user_movie["User_ID"] != user_id)
        ]["User_ID"].nunique()

        poster_html = (
            f'<img src="{m_img}" style="width:100%;height:100%;min-height:270px;object-fit:cover;display:block;" />'
            if m_img
            else '<div style="width:100%;min-height:270px;display:flex;align-items:center;justify-content:center;font-size:3rem;background:linear-gradient(135deg,#1a1040,#0d1b3e);">🎬</div>'
        )
        ur_val_s    = f"{user_rating_val:.1f}" if user_rating_val is not None else "N/A"
        m_avg_rat_s = f"{m_rating:.1f}" if not pd.isna(m_rating) else "N/A"

        st.markdown(f'''
        <div class="featured-card fade-in-up">
            <div class="featured-poster">{poster_html}</div>
            <div class="featured-body">
                <div class="featured-label">🎬 Recently Watched</div>
                <div class="featured-title">{m_title}</div>
                <div style="display:flex;gap:0.6rem;flex-wrap:wrap;align-items:center;font-size:0.8rem;color:var(--text-secondary);">
                    <span>📅 {m_year_s}</span>
                    <span style="width:4px;height:4px;border-radius:50%;background:var(--text-muted);display:inline-block;"></span>
                    <span>🎭 {m_genre}</span>
                    <span style="width:4px;height:4px;border-radius:50%;background:var(--text-muted);display:inline-block;"></span>
                    <span>🌐 {m_lang}</span>
                </div>
                <div style="color:var(--rating-color);font-size:1rem;">{m_stars} <span style="color:var(--text-secondary);font-size:0.8rem;">{m_rat_s}</span></div>
                <div style="display:flex;gap:1rem;flex-wrap:wrap;">
                    <div class="featured-stat"><div class="featured-stat-val">{ur_val_s}</div><div class="featured-stat-label">Your Rating</div></div>
                    <div class="featured-stat"><div class="featured-stat-val">{co_w}</div><div class="featured-stat-label">Co-Watchers</div></div>
                    <div class="featured-stat"><div class="featured-stat-val">{m_avg_rat_s}</div><div class="featured-stat-label">Avg Rating</div></div>
                </div>
                <div style="margin-top:auto;font-size:0.8rem;color:var(--text-muted);">
                    Based on {co_w} viewer(s) who also watched this, MovieMind found {len(recommendations)} personalized recommendation(s).
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)


# ============================================================
# STATS ROW
# ============================================================

def render_stats_row(recommendations):
    if not recommendations.empty:
        avg_rat  = recommendations["Rating"].mean() if "Rating" in recommendations.columns else 0
        genres_c = recommendations["Genre"].nunique() if "Genre" in recommendations.columns else 0
        langs_c  = recommendations["Language"].nunique() if "Language" in recommendations.columns else 0
        avg_rat_s = f"{avg_rat:.1f}" if avg_rat else "N/A"

        st.markdown(f'''
        <div class="stats-row fade-in-up">
            <div class="stat-card"><div class="stat-icon stat-icon-purple">✨</div><div><div class="stat-val">{len(recommendations)}</div><div class="stat-label">Recommendations</div></div></div>
            <div class="stat-card"><div class="stat-icon stat-icon-gold">⭐</div><div><div class="stat-val">{avg_rat_s}</div><div class="stat-label">Avg Rating</div></div></div>
            <div class="stat-card"><div class="stat-icon stat-icon-pink">🎭</div><div><div class="stat-val">{genres_c}</div><div class="stat-label">Genres</div></div></div>
            <div class="stat-card"><div class="stat-icon stat-icon-teal">🌐</div><div><div class="stat-val">{langs_c}</div><div class="stat-label">Languages</div></div></div>
        </div>
        ''', unsafe_allow_html=True)


# ============================================================
# PREMIUM MOVIE CARDS
# ============================================================

def render_premium_movie_cards(recommendations, user_id):
    if recommendations.empty:
        return

    all_genres = sorted(recommendations["Genre"].dropna().unique().tolist()) if "Genre" in recommendations.columns else []
    all_langs  = sorted(recommendations["Language"].dropna().unique().tolist()) if "Language" in recommendations.columns else []

    col_f1, col_f2, col_f3 = st.columns([3, 3, 2])
    with col_f1:
        genre_filter = st.selectbox("Genre", ["All Genres"] + all_genres, label_visibility="visible", key="genre_filter") if all_genres else "All Genres"
    with col_f2:
        lang_filter = st.selectbox("Language", ["All Languages"] + all_langs, label_visibility="visible", key="lang_filter") if all_langs else "All Languages"
    with col_f3:
        sort_by = st.selectbox("Sort By", ["AI Score", "Rating \u2193", "Rating \u2191", "Year \u2193", "Year \u2191"], label_visibility="visible", key="sort_filter")

    filtered = recommendations.copy()
    if genre_filter != "All Genres":
        filtered = filtered[filtered["Genre"] == genre_filter]
    if lang_filter != "All Languages":
        filtered = filtered[filtered["Language"] == lang_filter]
    if sort_by == "Rating \u2193":
        filtered = filtered.sort_values("Rating", ascending=False)
    elif sort_by == "Rating \u2191":
        filtered = filtered.sort_values("Rating", ascending=True)
    elif sort_by == "Year \u2193":
        filtered = filtered.sort_values("Release_Year", ascending=False)
    elif sort_by == "Year \u2191":
        filtered = filtered.sort_values("Release_Year", ascending=True)
    filtered = filtered.reset_index(drop=True)

    if filtered.empty:
        st.markdown('<div class="empty-state"><div class="empty-state-icon">\U0001f50d</div><div class="empty-state-title">No movies match this filter</div><div class="empty-state-desc">Try adjusting the genre or language filter.</div></div>', unsafe_allow_html=True)
        return

    cards_per_row = 5
    for start in range(0, len(filtered), cards_per_row):
        batch   = filtered.iloc[start:start + cards_per_row]
        columns = st.columns(cards_per_row, gap="small")

        for col, (local_idx, (_, movie)) in zip(columns, enumerate(batch.iterrows())):
            global_idx = start + local_idx
            with col:
                m_title   = str(movie["Movie_Title"]) if not pd.isna(movie.get("Movie_Title")) else "Unknown"
                m_year    = movie.get("Release_Year", None)
                m_genre   = str(movie.get("Genre", "Unknown")) if not pd.isna(movie.get("Genre", None)) else "Unknown"
                m_lang    = str(movie.get("Language", "Unknown")) if not pd.isna(movie.get("Language", None)) else "Unknown"
                m_rating  = movie.get("Rating", None)
                m_viewers = movie.get("Number_of_Viewers", movie.get("Direct_CoWatchers", None))
                m_img     = get_movie_image(m_title, m_year)
                m_stars   = render_stars(m_rating) if m_rating is not None and not pd.isna(m_rating) else "\u2606\u2606\u2606\u2606\u2606"
                m_year_s  = str(int(m_year)) if m_year is not None and not pd.isna(m_year) else "N/A"
                m_rat_s   = f"{m_rating:.1f}" if m_rating is not None and not pd.isna(m_rating) else "N/A"
                badge_html = get_badge(global_idx, m_rating, m_viewers)
                yt_query    = urllib.parse.quote(f"{m_title} official trailer")
                trailer_url = f"https://www.youtube.com/results?search_query={yt_query}"

                poster_content = (
                    f'<img src="{m_img}" style="width:100%;height:100%;object-fit:cover;display:block;" />'
                    if m_img
                    else '<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:2.5rem;color:rgba(139,92,246,0.3);">\U0001f3ac</div>'
                )
                viewers_html = (
                    f'<div class="movie-card-viewers">\U0001f465 {int(m_viewers)} viewer(s)</div>'
                    if m_viewers is not None and not pd.isna(m_viewers) else ""
                )
                heart_key     = f"heart_{global_idx}"
                is_wishlisted = m_title in st.session_state.wishlist
                heart_icon    = "\u2764\ufe0f" if is_wishlisted else "\U0001f90d"

                st.markdown(f"""
                <div class="movie-card fade-in-up">
                    <div class="movie-card-poster">
                        {poster_content}
                        <div class="movie-card-rank">#{global_idx + 1}</div>
                        <div style="position:absolute;top:0.5rem;right:0.5rem;width:32px;height:32px;border-radius:50%;background:rgba(5,8,22,0.75);backdrop-filter:blur(8px);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:0.9rem;">{heart_icon}</div>
                        <div class="movie-card-overlay">
                            <a href="{trailer_url}" target="_blank" class="movie-card-trailer-btn">\u25b6 Watch Trailer</a>
                        </div>
                    </div>
                    <div class="movie-card-body">
                        <div>{badge_html}</div>
                        <div class="movie-card-title">{m_title}</div>
                        <div class="movie-card-meta">
                            <span class="meta-pill">\U0001f3ad {m_genre}</span>
                            <span class="meta-pill">\U0001f4c5 {m_year_s}</span>
                            <span class="meta-pill">\U0001f310 {m_lang}</span>
                        </div>
                        <div class="movie-card-rating">
                            <span class="star-rating">{m_stars}</span>
                            <span class="rating-val">{m_rat_s}</span>
                        </div>
                        {viewers_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                btn_label = "\u2764\ufe0f Wishlisted" if is_wishlisted else "\U0001f90d Add to Watchlist"
                if st.button(btn_label, key=heart_key, use_container_width=True):
                    if is_wishlisted:
                        st.session_state.wishlist.discard(m_title)
                    else:
                        st.session_state.wishlist.add(m_title)
                    st.rerun()


# ============================================================


# ============================================================
# PAGE ROUTER
# ============================================================

active_page = st.session_state.active_page

# ── HOME / RECOMMENDATIONS ────────────────────────────────
if active_page in ("Home", "Recommendations"):

    if not recommendations.empty:
        render_hero(recommendations, user_id)
        if st.session_state.last_mode == "recent" and st.session_state.get("selected_movie_key"):
            render_featured_recent(user_id, st.session_state.selected_movie_key, recommendations)
        render_stats_row(recommendations)

        st.markdown(f'''
        <div style="display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:1.25rem;" class="fade-in-up">
            <div>
                <div class="section-title">✨ Movies You May Like</div>
                <div class="section-subtitle">Personalized recommendations • {len(recommendations)} result(s) for Viewer #{user_id}</div>
            </div>
            <span class="view-all-btn">View All →</span>
        </div>
        ''', unsafe_allow_html=True)
        render_premium_movie_cards(recommendations, user_id)
    else:
        total_movies_w = movie_details.shape[0]
        total_users_w  = user_movie["User_ID"].nunique()
        st.markdown(f'''
        <div class="welcome-hero fade-in-up">
            <div style="font-size:4rem;margin-bottom:1.25rem;">🎬</div>
            <div class="welcome-title">Your AI Cinematic Companion</div>
            <div class="welcome-desc">Discover movies you will love using advanced collaborative filtering AI.</div>
            <div style="display:flex;gap:0.75rem;justify-content:center;flex-wrap:wrap;margin-bottom:2rem;">
                <div style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);border-radius:99px;padding:0.4rem 1.1rem;font-size:0.8rem;color:var(--accent-highlight);font-weight:600;">{total_movies_w} Movies</div>
                <div style="background:rgba(217,70,239,0.08);border:1px solid rgba(217,70,239,0.2);border-radius:99px;padding:0.4rem 1.1rem;font-size:0.8rem;color:#E879F9;font-weight:600;">{total_users_w} Viewers</div>
                <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.2);border-radius:99px;padding:0.4rem 1.1rem;font-size:0.8rem;color:var(--rating-color);font-weight:600;">AI-Powered</div>
            </div>
            <div style="display:flex;gap:1.5rem;justify-content:center;flex-wrap:wrap;">
                <div class="welcome-step"><div class="welcome-step-num">01</div><div class="welcome-step-title">Choose a Mode</div><div class="welcome-step-desc">Pick By User or By Recently Watched in the sidebar</div></div>
                <div class="welcome-step"><div class="welcome-step-num">02</div><div class="welcome-step-title">Select Viewer</div><div class="welcome-step-desc">Choose a viewer profile to personalize picks</div></div>
                <div class="welcome-step"><div class="welcome-step-num">03</div><div class="welcome-step-title">Set Count</div><div class="welcome-step-desc">Use the slider to set how many recommendations</div></div>
                <div class="welcome-step"><div class="welcome-step-num">04</div><div class="welcome-step-title">Get Picks ✨</div><div class="welcome-step-desc">Click Get Recommendations and discover your next favorite</div></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# ── TRENDING ──────────────────────────────────────────────
elif active_page == "Trending":

    st.markdown("""
    <div class="fade-in-up" style="margin-bottom:1.25rem;">
        <div class="section-title">\U0001f525 Trending Now</div>
        <div class="section-subtitle">Top-rated movies across all viewers on the platform</div>
    </div>
    """, unsafe_allow_html=True)

    trending = (
        movies_data
        .groupby("Movie_Key")
        .agg(
            Movie_Title=("Movie_Title", "first"),
            Release_Year=("Release_Year", "first"),
            Genre=("Genre", "first"),
            Language=("Language", "first"),
            Rating=("Rating", "mean"),
            Rating_Count=("Rating", "count")
        )
        .reset_index()
    )
    trending = trending[trending["Rating_Count"] >= 3].sort_values("Rating", ascending=False).head(10).reset_index(drop=True)

    if trending.empty:
        st.info("Not enough rating data to show trending movies yet.")
    else:
        avg_t    = trending["Rating"].mean()
        genres_t = trending["Genre"].nunique()
        langs_t  = trending["Language"].nunique()
        avgt_s   = f"{avg_t:.1f}"
        st.markdown(f"""
        <div class="stats-row fade-in-up">
            <div class="stat-card"><div class="stat-icon stat-icon-gold">\U0001f3c6</div><div><div class="stat-val">{len(trending)}</div><div class="stat-label">Top Movies</div></div></div>
            <div class="stat-card"><div class="stat-icon stat-icon-purple">\u2b50</div><div><div class="stat-val">{avgt_s}</div><div class="stat-label">Avg Rating</div></div></div>
            <div class="stat-card"><div class="stat-icon stat-icon-pink">\U0001f3ad</div><div><div class="stat-val">{genres_t}</div><div class="stat-label">Genres</div></div></div>
            <div class="stat-card"><div class="stat-icon stat-icon-teal">\U0001f310</div><div><div class="stat-val">{langs_t}</div><div class="stat-label">Languages</div></div></div>
        </div>
        """, unsafe_allow_html=True)

        cards_per_row = 5
        for start_r in range(0, len(trending), cards_per_row):
            batch   = trending.iloc[start_r:start_r + cards_per_row]
            columns = st.columns(cards_per_row, gap="small")
            for col, (lidx, (_, movie)) in zip(columns, enumerate(batch.iterrows())):
                gidx = start_r + lidx
                with col:
                    mt   = str(movie["Movie_Title"]) if not pd.isna(movie.get("Movie_Title")) else "Unknown"
                    my   = movie.get("Release_Year", None)
                    mg   = str(movie.get("Genre", "Unknown")) if not pd.isna(movie.get("Genre", None)) else "Unknown"
                    ml   = str(movie.get("Language", "Unknown")) if not pd.isna(movie.get("Language", None)) else "Unknown"
                    mr   = movie.get("Rating", None)
                    mc   = int(movie.get("Rating_Count", 0))
                    mimg = get_movie_image(mt, my)
                    mst  = render_stars(mr) if mr is not None and not pd.isna(mr) else "\u2606\u2606\u2606\u2606\u2606"
                    mys  = str(int(my)) if my is not None and not pd.isna(my) else "N/A"
                    mrs  = f"{mr:.1f}" if mr is not None and not pd.isna(mr) else "N/A"
                    ytq  = urllib.parse.quote(f"{mt} official trailer")
                    turl = f"https://www.youtube.com/results?search_query={ytq}"
                    badge = "<span class='badge badge-gold'>\U0001f3c6 Top Rated</span>" if gidx == 0 else "<span class='badge badge-purple'>\U0001f525 Trending</span>"
                    pc   = f'<img src="{mimg}" style="width:100%;height:100%;object-fit:cover;display:block;" />' if mimg else '<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:2.5rem;">\U0001f3ac</div>'
                    st.markdown(f"""
                    <div class="movie-card fade-in-up">
                        <div class="movie-card-poster">
                            {pc}
                            <div class="movie-card-rank">#{gidx + 1}</div>
                            <div class="movie-card-overlay"><a href="{turl}" target="_blank" class="movie-card-trailer-btn">\u25b6 Watch Trailer</a></div>
                        </div>
                        <div class="movie-card-body">
                            <div>{badge}</div>
                            <div class="movie-card-title">{mt}</div>
                            <div class="movie-card-meta">
                                <span class="meta-pill">\U0001f3ad {mg}</span>
                                <span class="meta-pill">\U0001f4c5 {mys}</span>
                                <span class="meta-pill">\U0001f310 {ml}</span>
                            </div>
                            <div class="movie-card-rating"><span class="star-rating">{mst}</span><span class="rating-val">{mrs}</span></div>
                            <div class="movie-card-viewers">\U0001f465 {mc} rating(s)</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ── MY WATCHLIST ──────────────────────────────────────────
elif active_page == "My Watchlist":

    wl_count = len(st.session_state.wishlist)
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;" class="fade-in-up">
        <div class="section-title">\u2764\ufe0f My Watchlist</div>
        <div class="section-subtitle">{wl_count} movie(s) saved</div>
    </div>
    """, unsafe_allow_html=True)

    if wl_count == 0:
        st.markdown("""
        <div class="empty-state fade-in-up">
            <div class="empty-state-icon">\U0001f90d</div>
            <div class="empty-state-title">Your Watchlist is Empty</div>
            <div class="empty-state-desc">Get recommendations and click Add to Watchlist on any movie card to save it here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        wishlist_movies = sorted(st.session_state.wishlist)
        wl_cols = st.columns(min(wl_count, 5), gap="small")
        for wi, (wcol, wtitle) in enumerate(zip(wl_cols, wishlist_movies)):
            with wcol:
                wmatch = movie_details[normalize_title(movie_details["Movie_Title"]) == wtitle.strip().lower()]
                wy     = wmatch.iloc[0]["Release_Year"] if not wmatch.empty else None
                wg     = str(wmatch.iloc[0]["Genre"]) if not wmatch.empty and not pd.isna(wmatch.iloc[0]["Genre"]) else "Unknown"
                wr     = wmatch.iloc[0]["Rating"] if not wmatch.empty else None
                wrs    = f"{wr:.1f}" if wr is not None and not pd.isna(wr) else "N/A"
                wys    = str(int(wy)) if wy is not None and not pd.isna(wy) else "N/A"
                wimg   = get_movie_image(wtitle, wy)
                wpost  = f'<img src="{wimg}" style="width:100%;height:100%;object-fit:cover;display:block;" />' if wimg else '<div style="height:200px;display:flex;align-items:center;justify-content:center;font-size:2rem;background:linear-gradient(135deg,#1a1040,#0d1b3e);">\U0001f3ac</div>'
                wytq   = urllib.parse.quote(f"{wtitle} official trailer")
                wturl  = f"https://www.youtube.com/results?search_query={wytq}"
                st.markdown(f"""
                <div class="movie-card fade-in-up">
                    <div class="movie-card-poster">
                        {wpost}
                        <div class="movie-card-overlay"><a href="{wturl}" target="_blank" class="movie-card-trailer-btn">\u25b6 Watch Trailer</a></div>
                    </div>
                    <div class="movie-card-body">
                        <span class="badge badge-purple">\u2764\ufe0f Watchlisted</span>
                        <div class="movie-card-title">{wtitle}</div>
                        <div class="movie-card-meta">
                            <span class="meta-pill">\U0001f3ad {wg}</span>
                            <span class="meta-pill">\U0001f4c5 {wys}</span>
                        </div>
                        <div class="movie-card-rating"><span class="rating-val">\u2b50 {wrs}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("\U0001f5d1\ufe0f Remove", key=f"wl_remove_{wi}", use_container_width=True):
                    st.session_state.wishlist.discard(wtitle)
                    st.rerun()

# ── HISTORY ───────────────────────────────────────────────
elif active_page == "History":

    st.markdown("""
    <div style="margin-bottom:1.5rem;" class="fade-in-up">
        <div class="section-title">\U0001f550 Watch History</div>
        <div class="section-subtitle">Your most recent recommendation session</div>
    </div>
    """, unsafe_allow_html=True)

    if recommendations.empty:
        st.markdown("""
        <div class="empty-state fade-in-up">
            <div class="empty-state-icon">\U0001f550</div>
            <div class="empty-state-title">No Session History Yet</div>
            <div class="empty-state-desc">Go to Home, select a viewer and click Get Recommendations. Your session results will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        last_m = "By User" if st.session_state.last_mode == "user" else "By Recently Watched Movie"
        st.markdown(f"""
        <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);
                    border-radius:14px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;
                    display:flex;align-items:center;gap:1rem;" class="fade-in-up">
            <div style="font-size:2rem;">\U0001f4cb</div>
            <div>
                <div style="font-weight:700;color:var(--text-primary);font-size:0.95rem;">Last Session</div>
                <div style="font-size:0.82rem;color:var(--text-muted);margin-top:0.2rem;">
                    Mode: <span style="color:var(--accent-highlight);font-weight:600;">{last_m}</span>
                    &nbsp;\u2022&nbsp; Viewer:
                    <span style="color:var(--accent-highlight);font-weight:600;">#{user_id}</span>
                    &nbsp;\u2022&nbsp; Results:
                    <span style="color:var(--accent-highlight);font-weight:600;">{len(recommendations)} movie(s)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_premium_movie_cards(recommendations, user_id)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="margin-top:4rem;padding:2rem;text-align:center;border-top:1px solid var(--border);">
    <div style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;
                background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary));
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:0.5rem;">
        \U0001f3ac MovieMind
    </div>
    <div style="font-size:0.78rem;color:var(--text-muted);">
        AI-Powered Movie Discovery \u2022 Collaborative Filtering Engine \u2022 Built with Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
