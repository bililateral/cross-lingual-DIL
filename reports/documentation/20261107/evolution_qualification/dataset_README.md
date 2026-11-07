# A large-scale longitudinal structured dataset of the dark web cryptomarket Evolution (2014–2015)

Data includes a structured dataset of the forum and marketplace of the dark web cryptomarket Evolution as well as a longitudinal network snapshot dataset of a temporal weighted communication network extracted from the forum data. The datasets were extracted from the Dark Net Market archive data while resolving many of the data quality issues inherent to raw data scraped from online sources.
Dark web cryptomarkets facilitate the online trade of illicit goods. Evolution was active from January 2014 until March 2015; and formed a combination of  a carding forum and an underground drug market. This dataset includes the vast majority of data that could be obtained from the webpages of the forum and marketplace of Evolution. This includes, for example, information on the number of sales each individual vendor has completed, forum post contents, and product categories.
The extracted communication network models users' co-posting in the same forum topics, in essence modelling communication between them.

## Sharing/Access information

Links to other publicly accessible locations of the data:
 - N/A

Data was derived from the following sources:
 - [Gwern Branwen, Nicolas Christin, David Décary-Hétu, Rasmus Munksgaard Andersen, StExo, El Presidente, Anonymous, Daryl Lau, Sohhlz, Delyan Kratunov, Vince Cakic, Van Buskirk, Whom, Michael McKenna, Sigi Goode. “Dark Net Market archives, 2011–2015”, 2015-07-12. Web. [Accessed 2021-07-24]](https://gwern.net/dnm-archive)


## Description of the data and file structure

More extensive descriptions of the dataset, it's extraction, completeness, peculiarities, and some statistics are provided in this [Data Descriptor](https://doi.org/10.48550/arXiv.2311.11878).

### Forum data

#### "forum/scrapes.tsv":


| Variable        | Data type   | Description                                                          |
|-----------------|-------------|----------------------------------------------------------------------|
| scrape_id       | int         | forum scrape unique identifier                                       |
| scrape_year     | int         | year in which scrape was performed                                   |
| scrape_month    | int         | month in which scrape was performed                                  |
| scrape_day      | int         | day on which scrape was performed                                    |
| fora            | int         | number of fora at the time of the scrape as reported by index page   |
| topics          | int         | number of topics at the time of the scrape as reported by index page |
| posts           | int         | number of posts at the time of the scrape as reported by index page  |
| users           | int         | number of users at the time of the scrape as reported by index page  |

Note:

    * The statistics on 'fora', 'topics', 'posts' and 'users', are those reported on the index page
      of the forum. They are not a representation of what was actually gathered, and thus won't
      (necessarily) match the numbers extracted into this dataset.

#### "forum/forum.tsv":

| Variable        | Data type   | Description                                                          |
|-----------------|-------------|----------------------------------------------------------------------|
| fid             | int         | forum unique identifier                                              |
| scrape_id       | int         | forum scrape unique identifier (foreign key)                         |
| category        | string      | forum category to which this forum belongs                           |
| title           | string      | forum title                                                          |
| description     | string      | forum description                                                    |
| pages           | int         | number of pages with topics in this forum, as implied by forum pages |
| topics          | int         | number of (expected) topics in this forum, as reported by statistics on index page or implied by forum pages |
| topics_visible  | int         | number of unique topics actually found through forum pages alone     |
| topics_found    | int         | number of unique topics associated with this forum found through either forum or topic pages in all scrapes so far |
| posts           | int         | number of (expected) posts in this forum, as reported by index page  |
| posts_found     | int         | number of unique posts with a placement date up to and including the current scrape in topics counted towards topics_found |

Note:

    * 'topics_found'/'posts_found' may include those deleted by Moderators and exceed 'topics'/'posts'

#### "forum/topic.tsv":

| Variable        | Data type   | Description                                                             |
|-----------------|-------------|-------------------------------------------------------------------------|
| fid             | int         | forum unique identifier (foreign key) to which the topic belongs or belonged (if moved is True) |
| tid             | int         | topic unique identifier                                                 |
| first_uid       | int         | user unique identifier (foreign key) of the forum user that placed the first post of the topic |
| scrape_id       | int         | forum scrape unique identifier (foreign key)                            |
| title           | string      | topic title                                                             |
| posts           | int         | number of posts in this topic, as reported by statistics on forum pages or implied by topic pages |
| posts_visible   | int         | number of unique posts actually found on topic pages                    |
| posts_found     | int         | number of unique posts found in all scrapes so far, with placement dates up to and including the scrape date |
| views           | int         | number of views of this topic, as reported by statistics on forum pages |
| lp_uid          | int         | user unique identifier (foreign key) of user who placed the last post, as reported on forum pages |
| lp_year         | int         | year of last post, as reported on forum pages                           |
| lp_month        | int         | month of last post, as reported on forum pages                          |
| lp_day          | int         | day of last post, as reported on forum pages                            |
| lp_time         | time        | time of last post, as reported on forum pages                           |
| closed          | bool        | boolean flag indicating whether this topic was closed                   |
| moved           | bool        | boolean flag indicating whether this topic was moved to another forum (fid) |

Note:

    * A topic entry can exist for each forum (fid) in which it is found within a given scrape.
      Whether it is still officially part of said scrape is indicated with the 'moved' variable.
      If a topic has been moved: posts, views and last post information is copied from the
      forum where it was moved to (when available)

    * Reported last post details may not match the last post found in "post.tsv". The stored
      details match those (latest) reported on the forum pages and are not otherwise resolved.


#### "forum/post.tsv":

| Variable        | Data type   | Description                                                                         |
|-----------------|-------------|-------------------------------------------------------------------------------------|
| tid             | int         | topic unique identifier (foreign key), indicating to which topic this post belongs  |
| pid             | int         | post unique identifier                                                              |
| seq_id          | int         | sequence id, indicating where in the order of posts of the topic this post belongs  |
| year            | int         | year that the post was placed                                                       |
| month           | int         | month that the post was placed                                                      |
| day             | int         | day that the post was placed                                                        |
| time            | time        | time that the post was placed                                                       |
| uid             | int         | user unique identifier (foreign key) of the user who placed the post                |
| text            | string      | the post’s message text, as directly copied from source topic pages                 |
| signature       | string      | signature of the user who placed the post (most recent found for this post)         |
| edit_uid        | int         | user unique identifier (foreign key) of the user who last edited the post           |
| edit_year       | int         | year of the last edit to the post                                                   |
| edit_month      | int         | month of the last edit to the post                                                  |
| edit_day        | int         | day of the last edit to the post                                                    |
| edit_time       | time        | time of the last edit to the post                                                   |

Note:

    * For posts we only kept their final form, i.e., the text after it was last edited. Earlier
      versions were not stored as we had no interest in analysing textual changes upon edits.
      Additionally, not including every version of post text saved on space.

    * 'seq_id' is not determined by that retrieved from the scrape, but based on all posts we were
      able to retrieve. In other words, any posts that were not obtained in ANY scrape are also
      excluded from this ordering and thus the 'seq_id' pretends those posts did not exist at all.
      In total we observed 113 gaps in 65 topics, with a cumulative size of 3525 missing posts.

    * signature and edit information can be missing, however this is always due to it simply not
      existing, not due to any extraction error.

#### "forum/user.tsv":

| Variable        | Data type   | Description                                                                                 |
|-----------------|-------------|---------------------------------------------------------------------------------------------|
| uid             | int         | user unique identifier                                                                      |
| username        | string      | username of user                                                                            |
| reg_year        | int         | registration year of user                                                                   |
| reg_month       | int         | registration month of user                                                                  |
| reg_day         | int         | registration day of user                                                                    |
| scrape_id       | int         | forum scrape unique identifier (foreign key)                                                |
| title           | string      | member title of user, indicating their role on the forum                                    |
| lp_year         | int         | year of last post placement, as reported on profile page                                    |
| lp_month        | int         | month of last post placement, as reported on profile page                                   |
| lp_day          | int         | day of last post placement, as reported on profile page                                     |
| lp_time         | time        | time of last post placement, as reported on profile page                                    |
| num_posts       | int         | number of posts by user to date, as reported on profile page or alongside the user’s posts  |
| location        | string      | self-reported location of forum user, highly unreliable                                     |

Note:

    * Location is often missing or fake, for example, "end of the world" or "Legio XIII Gemina".

    * When there is no information other than the uid, username and scrape, this user placed no
      posts (within our data) and had no profile file, but either showed up as an editor or as
      an initial or final poster on a topic listed on a forum page (but whose posts were thus
      not included in the scraped data).

    * There may be an apparent mismatch between the 'num_posts' and last post information when
      considered over time, e.g., last post information may change whilst 'num_posts' does not.
      This is due to the post count potentially having been retrieved from a file on a later date
      than the profile file with the last post information was. As such, the post count can
      already include newer posts than the apparent last post.

    * On user title conflicts the most significant title was stored based on chosen ordering:
      Administrator > Market Moderator > Forum Moderator > Moderator > Public Relations > Banned >
      Vendor > Resident Medical Expert > Troll > Member > Guest > Sports Referee > Sports Fan

    * On 38 occasions two or three uid's have the same username. Inspection of data suggests
      these are cases of a user's account being banned and them starting a new account.


### Market data

#### "market/scrapes.tsv":

| Variable          | Data type | Description                                                     |
|-------------------|-----------|-----------------------------------------------------------------|
| mscrape_id        | int       | market scrape unique identifier                                 |
| scrape_year       | int       | year in which scrape was completed                              |
| scrape_month      | int       | month in which scrape was completed                             |
| scrape_day        | int       | day on which scrape was completed                               |

#### "market/categories.tsv":

| Variable          | Data type | Description                                                       |
|-------------------|-----------|-------------------------------------------------------------------|
| cid               | int       | product category unique identifier                                |
| category          | string    | name of category                                                  |
| parent_cid        | int       | product category unique identifier of parent category of category |

#### "market/listings.tsv":

| Variable          | Data type | Description                                                           |
|-------------------|-----------|-----------------------------------------------------------------------|
| lid               | int       | listing (unique) identifier                                           |
| vid               | int       | vendor unique identifier (foreign key)                                |
| mscrape_id        | int       | market scrape unique identifier (foreign key)                         |
| title             | string    | listing title, often includes basic description of product            |
| price             | string    | price in BTC per unit of listing                                      |
| description       | string    | description of product                                                |
| cid               | int       | product category unique identifier (foreign key)                      |
| ships_from        | string    | indication of from what part of the world the product can be shipped  |
| ships_to          | string    | indication of to which parts of the world product can be shipped      |
| product_class     | string    | indication of type of products (either Digital or Physical)           |
| listing_available | bool      | indication (boolean) of whether product of listing remains available  |
| return_policy     | string    | return policy associated with the listing                             |

Note:

    * For early scrapes, when description from a listing page not available, a shorter description
      from store/category pages may be stored as description instead.

    * At times the lid reported on store/category pages was not quite reliable. The cases where
      this led to conflicting vid were manually corrected. There appear to be more cases with the
      same vid (repeating exact title and information over multiple lid). However, as the raw source
      data at times presents these as also different listing pages, it is impossible to tell in
      which cases they are truly different listings and when it is abusively assigned to the wrong
      lid. Thus these cases are left unresolved!

    * Regularly a title change indicates an entirely new product being sold under the same lid.
      As such one should be mindful of this (and our previous point) when computing listing
      statistics.

    * Some titles have trailing spaces

#### "market/vendors.tsv":

| Variable          | Data type | Description                                                                                             |
|-------------------|-----------|---------------------------------------------------------------------------------------------------------|
| vid               | int       | vendor unique identifier                                                                                |
| mscrape_id        | int       | market scrape unique identifier (foreign key)                                                           |
| username          | string    | username of vendor                                                                                      |
| rank              | string    | rank of vendor based on combination of number of sales, feedback received and total price of all sales  |
| sales             | int       | number of sales vendor has reportedly completed so far (only available from market scrape 13 onwards)   |
| approval_rating   | float     | rating computed as the proportion of positive and negative feedback received over all the vendor’s listings that is positive |
| positive_feedback | int       | number of positive feedback responses the vendors has received on their listings                        |
| neutral_feedback  | int       | number of neutral feedback responses the vendors has received on their listings                         |
| negative_feedback | int       | number of negative feedback responses the vendors has received on their listings                        |
| legacy_sales      | string    | self-reported and ‘verified’ sales of vendor on other Dark Net cryptomarket(s)                          |
| pgp_key           | string    | PGP-key used by vendor for secure communication with customers                                          |
| return_policy     | string    | return policy of vendor                                                                                 |
| disabled          | bool      | indication (True or False) of whether the vendor account has been disabled                              |

Note:

    * Number of sales is only available starting from market scrape 13 (2014-05-10) when a change
      in the ranking system occurred. As such, this field is always empty for earlier scrapes.
      Before market scrape 13, the rank ordering was as follows: Freshman < Sophomore < Junior <
      Senior < Premium < Advanced < Expert < Master < Grandmaster < Godlike.

    * On 15 occasions two vid's have the same username. Inspection of data suggests these are
      cases of a user's account being banned and them starting a new account.

    * Exceptions exist for Gwern and Simurgh (the users who gathered the data). For these vendors
      only the 'vid', 'mscrape_id', 'username' are listed with 'disabled' set to False.

    * Sales counter can go down. Reason for this is unclear, perhaps has to do with cancelled
      orders, due to non-arrival. After all, purchases could be disputed by the customer.

#### "market/listing-feedback.tsv":

| Variable        | Data type   | Description                                                           |
|-----------------|-------------|-----------------------------------------------------------------------|
| lid             | int         | listing unique identifier (foreign key) to which feedback was placed  |
| username        | string      | username of the one who posted the feedback                           |
| year            | int         | year that feedback was placed                                         |
| month           | int         | month that feedback was placed                                        |
| day             | int         | day that feedback was placed                                          |
| message         | string      | feedback message                                                      |

Note:

    * Only 27 entries have a full username while remaining entries have a hidden username with only the
      first and last letter of the username visible.

    * Duplicate entries were deleted

    * This covers only feedback retrieved from the listing pages themselves. Feedback from vendor profile
      pages was not extracted.


### Forum user and market vendor matching

#### "forum-market/user-matching.tsv":

| Variable        | Data type   | Description                                                         |
|-----------------|-------------|---------------------------------------------------------------------|
| match_id        | int         | successful username match unique identifier                         |
| username        | string      | username upon which forum user(s) and market vendor(s) were matched |
| uid             | int         | forum user unique identifier (foreign key)                          |
| vid             | int         | market vendor unique identifier (foreign key)                       |


Note:

    * For those usernames without a successful match, entries with empty match_id are included

    * Multiple entries for the same match_id may exist for cases where multiple 'uid' or 'vid' had the
      same username. Note that these cases are associated with users being banned and starting a
      new account.


### Communication network(s)

The extracted networks model co-posting in the same topic. Specifically, an edge is formed if two users posted within ten posts and within one month of one another in the same topic. Additionally, an edge is formed from each user posting in a topic to the user who posted the initial post of a topic. For more details on the network (snapshot) extraction procedure we refer to this [Data Descriptor](TODO: link to data paper here!).

#### "network/nodes.tsv":

| Variable        | Data type   | Description                                                               |
|-----------------|-------------|---------------------------------------------------------------------------|
| uid             | int         | node identifier corresponding to forum user unique identifier (foreign key), lowest option if multiple options exist for the given username |
| secondary_uid   | int         | second uid associated with the same username (foreign key)                |
| tertiary_uid    | int         | third (and highest) uid associated with the same username (foreign key)   |
| match_id        | int         | successful username match unique identifier (foreign key)                 |
| init_year       | int         | first year that the user placed a post on the forum                       |
| init_month      | int         | first month (within init_year) that the user placed a post on the forum   |

#### "network/edges-2014-1.tsv", "network/edges-2014-2.tsv", ..., "network/edges-2015-3.tsv":

| Variable        | Data type   | Description                                                                                          |
|-----------------|-------------|------------------------------------------------------------------------------------------------------|
| Source          | int         | source node forum user unique identifier (foreign key)                                               |
| Target          | int         | target node forum user unique identifier (foreign key)                                               |
| Weight          | float       | weight of edge, as computed based on time passed                                                     |
| to_first        | bool        | boolean indication whether this edge is based on linking to initial post, if True, Weight set to 0.5 |
| time_diff       | int         | elapsed time in-between the posts that generated this edge, expressed in seconds                     |
| seq_diff        | int         | difference in seq_id between the posts that generated this edge                                      |
| timestamp       | int         | timestamp of source post, expressed in seconds since January 1st 2014                                |
| tid             | int         | topic unique identifier (foreign key), wherein the posts that generated this edge were placed        |