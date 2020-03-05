## Notes for Assignment 2

### A Universal Part-of-Speech Tagset [[source]](https://www.nltk.org/book/ch05.html#tab-universal-tagset)

Tagged corpora use many different conventions for tagging words. To help us get started, we will be looking at a simplified tagset.

Universal Part-of-Speech Tagset

| Tag | Meaning | English Examples |
| - | - | - |
| ADJ | adjective | new, good, high, special, big, local |
| ADP | adposition | on, of, at, with, by, into, under |
| ADV | adverb | really, already, still, early, now |
| CONJ | conjunction | and, or, but, if, while, although |
| DET | determiner, article | the, a, some, most, every, no, which |
| NOUN | noun | year, home, costs, time, Africa |
| NUM | numeral | twenty-four, fourth, 1991, 14:24 |
| PRT | particle | at, on, out, over per, that, up, with |
| PRON | pronoun | he, their, her, its, my, I, us |
| VERB | verb | is, say, told, given, playing, would |
| . | punctuation marks | . , ; ! |
| X | other | ersatz, esprit, dunno, gr8, univeristy |

### Nouns

`Nouns` generally refer to __people, places, things, or concepts__, e.g.: woman, Scotland, book, intelligence. Nouns can appear after `determiners` and `adjectives`, and can be the `subject` or `object` of the `verb`.

Syntactic Patterns involving some Nouns

| Word | After a determiner | Subject of the verb |
| - | - | - |
| woman | _the_ woman who I saw yesterday ... | the woman _sat_ down |
| Scotland | _the_ Scotland I remember as a child ... | Scotland _has_ five million people |
| book | _the_ book I bought yesterday ... | this book _recounts_ the colonization of Australia |
| intelligence | _the_ intelligence displayed by the child ... | Mary's intelligence _impressed_ her teachers |

The simplified noun tags are N for common nouns like _book_, and NP for proper nouns like _Scotland_.

### Verbs

`Verbs` are words that describe events and actions, e.g. _fall_, _eat_. In the context of a sentence, verbs typically express a relation involving the referents of one or more noun phrases.

| Word | Simple | With modifiers and adjuncts (italicized) |
| - | - | - |
| fall | Rome fell | Dot com stocks _suddenly_ fell _like a stone_ |
| eat | Mice eat cheese | John ate the pizza with _gusto_ |

### Adjectives and Adverbs

Two other important word classes are __adjectives__ and __adverbs__. __Adjectives describe nouns__, and can be used as __modifiers__ (e.g. in the _large_ pizza), or in predicates (e.g. the pizza is _large_). English adjectives can have internal structure (e.g. fall+ing in the _falling_ stocks). Adverbs modify verbs to specify the __time, manner, place or direction of the event__ described by the verb (e.g. the stocks fell _quickly_). Adverbs may also modify adjectives (e.g. Mary's teacher was _really_ nice).

English has several categories of closed class words in addition to prepositions, such as __articles__ (also often called __determiners__) (e.g., _the_, _a_), __modals___ (e.g., _should_, _may_), and __personal pronouns__ (e.g., _she_, _they_). Each dictionary and grammar classifies these words differently.