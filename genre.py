
class Genre():
    SCI_FI = "sci_fi"
    CRIME = "crime"
    ROMANCE = "romance"
    ANIMATION = "animation"
    MUSIC = "music"
    ADULT = "adult"
    COMEDY = "comedy"
    WAR = "war"
    HORROR = "horror"
    FILM_NOIR = "film_noir"
    WESTERN = "western"
    NEWS = "news"
    REALITY_TV = "reality_tv"
    THRILLER = "thriller"
    ADVENTURE = "adventure"
    MYSTERY = "mystery"
    SHORT = "short"
    TALK_SHOW = "talk_show"
    DRAMA = "drama"
    ACTION = "action"
    DOCUMENTARY = "documentary"
    MUSICAL = "musical"
    HISTORY = "history"
    FAMILY = "family"
    FANTASY = "fantasy"
    SPORT = "sport"
    BIOGRAPHY = "biography"


    ALL_GENRE = [SCI_FI, CRIME, ROMANCE, ANIMATION, MUSIC, ADULT, COMEDY, WAR, HORROR, FILM_NOIR, WESTERN, NEWS, REALITY_TV, THRILLER, ADVENTURE, MYSTERY, SHORT, TALK_SHOW, DRAMA, ACTION, DOCUMENTARY, MUSICAL, HISTORY, FAMILY, FANTASY, SPORT, BIOGRAPHY]

    @staticmethod
    def norm_genre(genre):
        return genre.lower().replace("-", "_") if genre else genre