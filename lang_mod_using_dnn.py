import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

data_text = """Titanic, in full Royal Mail Ship (RMS) Titanic, British luxury passenger liner that sank on April 14–15, 1912, during its maiden voyage, en route to New York City from Southampton, England, killing about 1,500 (see Researcher’s Note: Titanic) passengers and ship personnel. One of the most famous tragedies in modern history, it inspired numerous stories, several films, and a musical and has been the subject of much scholarship and scientific speculation.


Origins and construction
Titanic
Titanic
What if the Titanic hadn''t sunk?
What if the Titanic hadn''t sunk?See all videos for this article
J. Bruce Ismay
J. Bruce Ismay
William James Pirrie, Viscount Pirrie
William James Pirrie, Viscount Pirrie
In the early 1900s the transatlantic passenger trade was highly profitable and competitive, with ship lines vying to transport wealthy travelers and immigrants. Two of the chief lines were White Star and Cunard. By the summer of 1907, Cunard seemed poised to increase its share of the market with the debut of two new ships, the Lusitania and the Mauretania, which were scheduled to enter service later that year. The two passenger liners were garnering much attention for their expected speed; both would later set speed records crossing the Atlantic Ocean. Looking to answer his rival, White Star chairman J. Bruce Ismay reportedly met with William Pirrie, who controlled the Belfast shipbuilding firm Harland and Wolff, which constructed most of White Star’s vessels. The two men devised a plan to build a class of large liners that would be known for their comfort instead of their speed. It was eventually decided that three vessels would be constructed: the Olympic, the Titanic, and the Britannic.

construction of the ships Olympic and Titanic
construction of the ships Olympic and Titanic
Titanic: propellers
Titanic: propellers
Thomas Andrews
Thomas Andrews
On March 31, 1909, some three months after work began on the Olympic, the keel was laid for the Titanic. The two ships were built side by side in a specially constructed gantry that could accommodate their unprecedented size. The sister ships were largely designed by Thomas Andrews of Harland and Wolff. In addition to ornate decorations, the Titanic featured an immense first-class dining saloon, four elevators, and a swimming pool. Its second-class accommodations were comparable to first-class features on other ships, and its third-class offerings, although modest, were still noted for their relative comfort.

Newspaper boy Ned Parfett sells copies of the Evening News telling of the Titanic maritime disaster, outside Oceanic House, the London offices of the Titanic's owner, the White Star Line, in Cockspur Street, London, April 16, 1912.
Britannica Quiz
Understanding the Titanic Disaster
As to safety elements, the Titanic had 16 compartments that included doors which could be closed from the bridge, so that water could be contained in the event the hull was breached. Although they were presumed to be watertight, the bulkheads were not capped at the top. The ship’s builders claimed that four of the compartments could be flooded without endangering the liner’s buoyancy. The system led many to claim that the Titanic was unsinkable.



Explore the Titanic in an interactive diagram.
Titanic's Grand Staircase
Titanic's Grand Staircase
Titanic: parlour suite
Titanic: parlour suite
Titanic: gymnasium
Titanic: gymnasium
Following completion of the hull and main superstructure, the Titanic was launched on May 31, 1911. It then began the fitting-out phase, as machinery was loaded into the ship and interior work began. After the Olympic’s maiden voyage in June 1911, slight changes were made to the Titanic’s design. In early April 1912 the Titanic underwent its sea trials, after which the ship was declared seaworthy.


Get a Britannica Premium subscription and gain access to exclusive content.
Titanic: first-class lounge
Titanic: first-class lounge
Titanic: first-class dining saloon
Titanic: first-class dining saloon
As it prepared to embark on its maiden voyage, the Titanic was one of the largest and most opulent ships in the world. It had a gross registered tonnage (i.e., carrying capacity) of 46,328 tons, and when fully laden the ship displaced (weighed) more than 52,000 tons. The Titanic was approximately 882.5 feet (269 metres) long and about 92.5 feet (28.2 metres) wide at its widest point.

Maiden voyage
Watch actual footage of the Titanic
Watch actual footage of the TitanicSee all videos for this article
poster of the Titanic
poster of the Titanic
Edward J. Smith
Edward J. Smith
Isidor Straus
Isidor Straus
On April 10, 1912, the Titanic set sail on its maiden voyage, traveling from Southampton, England, to New York City. Nicknamed the “Millionaire’s Special,” the ship was fittingly captained by Edward J. Smith, who was known as the “Millionaire’s Captain” because of his popularity with wealthy passengers. Indeed, onboard were a number of prominent people, including American businessman Benjamin Guggenheim, British journalist William Thomas Stead, and Macy’s department store co-owner Isidor Straus and his wife, Ida. In addition, Ismay and Andrews were also traveling on the Titanic.


Titanic leaving Queenstown, Ireland
Titanic leaving Queenstown, Ireland
John Jacob Astor
John Jacob Astor
Molly Brown
Molly Brown
The voyage nearly began with a collision, however, when suction from the Titanic caused the docked New York to swing into the giant liner’s path. After an hour of maneuverings to prevent the accident, the Titanic was under way. On the evening of April 10 the ship stopped at Cherbourg, France. The city’s dock was too small to accommodate the Titanic, so passengers had to be ferried to and from the ship in tenders. Among those boarding were John Jacob Astor and his pregnant second wife, Madeleine, and Molly Brown. After some two hours the Titanic resumed its journey. On the morning of April 11 the liner made its last scheduled stop in Europe, at Queenstown (Cobh), Ireland. At approximately 1:30 PM the ship set sail for New York City. Onboard were some 2,200 people, approximately 1,300 of whom were passengers.

Final hours
reproduction of the Titanic's wireless room
reproduction of the Titanic's wireless room
Throughout much of the voyage, the wireless radio operators on the Titanic, Jack Phillips and Harold Bride, had been receiving iceberg warnings, most of which were passed along to the bridge. The two men worked for the Marconi Company, and much of their job was relaying passengers’ messages. On the evening of April 14 the Titanic began to approach an area known to have icebergs. Smith slightly altered the ship’s course to head farther south. However, he maintained the ship’s speed of some 22 knots. At approximately 9:40 PM the Mesaba sent a warning of an ice field. The message was never relayed to the Titanic’s bridge. At 10:55 PM the nearby Leyland liner Californian sent word that it had stopped after becoming surrounded by ice. Phillips, who was handling passenger messages, scolded the Californian for interrupting him.


Two lookouts, Frederick Fleet and Reginald Lee, were stationed in the crow’s nest of the Titanic. Their task was made difficult by the fact that the ocean was unusually calm that night: because there would be little water breaking at its base, an iceberg would be more difficult to spot. In addition, the crow’s nest’s binoculars were missing. At approximately 11:40 PM, about 400 nautical miles (740 km) south of Newfoundland, Canada, an iceberg was sighted, and the bridge was notified. First Officer William Murdoch ordered both the ship “hard-a-starboard”—a maneuver that under the order system then in place would turn the ship to port (left)—and the engines reversed. The Titanic began to turn, but it was too close to avoid a collision. The ship’s starboard side scraped along the iceberg. At least five of its supposedly watertight compartments toward the bow were ruptured. After assessing the damage, Andrews determined that, as the ship’s forward compartments filled with water, its bow would drop deeper into the ocean, causing water from the ruptured compartments to spill over into each succeeding compartment, thereby sealing the ship’s fate. The Titanic would founder. (By reversing the engines, Murdoch actually caused the Titanic to turn slower than if it had been moving at its original speed. Most experts believe the ship would have survived if it had hit the iceberg head-on.)

Smith ordered Phillips to begin sending distress signals, one of which reached the Carpathia at approximately 12:20 AM on April 15, and the Cunard ship immediately headed toward the stricken liner. However, the Carpathia was some 58 nautical miles (107 km) away when it received the signal, and it would take more than three hours to reach the Titanic. Other ships also responded, including the Olympic, but all were too far away. A vessel was spotted nearby, but the Titanic was unable to contact it. The Californian was also in the vicinity, but its wireless had been turned off for the night.

Titanic sinking
Titanic sinking
As attempts were made to contact nearby vessels, the lifeboats began to be launched, with orders of women and children first. Although the Titanic’s number of lifeboats exceeded that required by the British Board of Trade, its 20 boats could carry only 1,178 people, far short of the total number of passengers. This problem was exacerbated by lifeboats being launched well below capacity, because crewmen worried that the davits would not be able to support the weight of a fully loaded boat. (The Titanic had canceled its scheduled lifeboat drill earlier in the day, and the crew was unaware that the davits had been tested in Belfast.) Lifeboat number 7, which was the first to leave the Titanic, held only about 27 people, though it had space for 65. In the end, only 705 people would be rescued in lifeboats.


As passengers waited to enter lifeboats, they were entertained by the Titanic’s musicians, who initially played in the first-class lounge before eventually moving to the ship’s deck. Sources differ on how long they performed, some reporting that it was until shortly before the ship sank. Speculation also surrounded the last song they performed—likely either Autumn or Nearer My God to Thee. None of the musicians survived the sinking.

By 1:00 AM water was seen at the base (E deck) of the Grand Staircase. Amid the growing panic, several male passengers tried to board lifeboat number 14, causing Fifth Officer Harold Lowe to fire his gun three times. Around this time, Phillips’s distress calls reflected a growing desperation as one noted that the ship “cannot last much longer.”

As the Titanic’s bow continued to sink, the stern began to rise out of the water, placing incredible strain on the midsection. At about 2:00 AM the stern’s propellers were clearly visible above the water, and the only lifeboats that remained on the ship were three collapsible boats. Smith released the crew, saying that “it’s every man for himself.” (He was reportedly last seen in the bridge, and his body was never found.) At approximately 2:18 AM the lights on the Titanic went out. It then broke in two, with the bow going underwater. Reports later speculated that it took some six minutes for that section, likely traveling at approximately 30 miles (48 km) per hour, to reach the ocean bottom. The stern momentarily settled back in the water before rising again, eventually becoming vertical. It briefly remained in that position before beginning its final plunge. At 2:20 AM the ship foundered as the stern also disappeared beneath the Atlantic. Water pressure allegedly caused that section, which still had air inside, to implode as it sank.

Titanic survivors
Titanic survivors
Hundreds of passengers and crew went into the icy water. Fearful of being swamped, those in the lifeboats delayed returning to pick up survivors. By the time they rowed back, almost all the people in the water had died from exposure. In the end, more than 1,500 perished. Aside from the crew, which had about 700 fatalities, third class suffered the greatest loss: of approximately 710, only some 174 survived. (Subsequent claims that passengers in steerage were prevented from boarding boats, however, were largely dispelled. Given Smith’s failure to sound a general alarm, some third-class passengers did not realize the direness of the situation until it was too late. Many women also refused to leave their husbands and sons, while the difficulty of simply navigating the complex Titanic from the lower levels caused some to reach the top deck after most of the lifeboats had been launched.)


Read our timeline of the Titanic’s final hours.

Rescue
Titanic survivors
Titanic survivors
Titanic survivors aboard the Carpathia
Titanic survivors aboard the Carpathia
news of the Titanic's sinking
news of the Titanic's sinking
The Carpathia arrived in the area at approximately 3:30 AM, more than an hour after the Titanic sank. Lifeboat number 2 was the first to reach the liner. Over the next several hours the Carpathia picked up all survivors. White Star chairman Ismay wrote a message to be sent to the White Star Line’s offices: “Deeply regret advise you Titanic sank this morning fifteenth after collision iceberg, resulting serious loss life; further particulars later.” At approximately 8:30 AM the Californian arrived, having heard the news some three hours earlier. Shortly before 9:00 AM the Carpathia headed for New York City, where it arrived to massive crowds on April 18.

Aftermath and investigation
Study the causes of and fallout from the Titanic's striking an iceberg and sinking in the Atlantic Ocean
Study the causes of and fallout from the Titanic's striking an iceberg and sinking in the Atlantic OceanSee all videos for this article
Carpathia Capt. Arthur Henry Rostron and Molly Brown
Carpathia Capt. Arthur Henry Rostron and Molly Brown
Although the majority of dead were crew members and third-class passengers, many of the era’s wealthiest and most prominent families lost members, among them Isidor and Ida Straus and John Jacob Astor. In the popular mind, the glamour associated with the ship, its maiden voyage, and its notable passengers magnified the tragedy of its sinking. Legends arose almost immediately about the night’s events, those who had died, and those who survived. Heroes and heroines—such as American Molly Brown, who helped command a lifeboat, and Capt. Arthur Henry Rostron of the Carpathia—were identified and celebrated by the press. Others—notably Ismay, who had found space in a lifeboat and survived—were vilified. There was a strong desire to explain the tragedy, and inquiries into the sinking were held in the United States and Great Britain.

U.S. inquiry
U.S. Senate investigation of the Titanic's sinking
U.S. Senate investigation of the Titanic's sinking
The U.S. investigation, which lasted from April 19 to May 25, 1912, was led by Sen. William Alden Smith. In all, more than 80 people were interviewed. Notable witnesses included Second Officer Charles Lightoller, the most senior officer to survive. He defended the actions of his superiors, especially Captain Smith’s refusal to decrease the ship’s speed. Many passengers testified to the general confusion on the ship. A general warning was never sounded, causing a number of passengers and even crew members to be unaware of the danger for some time. In addition, because a scheduled lifeboat drill had never been held, the lowering of the boats was often haphazard.

Perhaps the most-scrutinized testimony came from the crew of the Californian, who claimed their ship was some 20 nautical miles (37 km) from the Titanic. Crew members saw a ship but said it was too small to be the Titanic. They also stated that it was moving and that efforts to contact it by Morse lamp were unsuccessful. After sighting rockets in the distance, the crew informed Capt. Stanley Lord, who had retired for the night. Instead of ordering the ship’s wireless operator to turn on the radio, Lord instead told the men to continue to use the Morse lamp. By 2:00 AM the nearby ship had reportedly sailed away.

In the end, the U.S. investigation faulted the British Board of Trade, “to whose laxity of regulation and hasty inspection the world is largely indebted for this awful fatality.” Other contributing causes were also noted, including the failure of Captain Smith to slow the Titanic after receiving ice warnings. However, perhaps the strongest criticism was levied at Captain Lord and the Californian. The committee found that the ship was “nearer the Titanic than the 19 miles reported by her Captain, and that her officers and crew saw the distress signals of the Titanic and failed to respond to them in accordance with the dictates of humanity, international usage, and the requirements of law.”

British inquiry
liability claim of a Titanic survivor
liability claim of a Titanic survivor
In May 1912 the British inquiry began. It was overseen by the British Board of Trade, the same agency that had been derided by U.S. investigators for the insufficient lifeboat requirements. The presiding judge was Sir John Charles Bigham, Lord Mersey. Little new evidence was discovered during the 28 days of testimony. The final report stated that “the loss of the said ship was due to collision with an iceberg, brought about by the excessive speed at which the ship was being navigated.” However, Mersey also stated that he was “not able to blame Captain Smith…he was doing only that which other skilled men would have done in the same position.” Captain Lord and the Californian, however, drew sharp rebuke. The British investigators claimed that the liner was some 5–10 nautical miles (9–19 km) from the Titanic and that “she might have saved many, if not all, of the lives that were lost.”


Both the U.S. and British investigations also proposed various safety recommendations, and in 1913 the first International Conference for Safety of Life at Sea was called in London. The conference drew up rules requiring that every ship have lifeboat space for each person embarked; that lifeboat drills be held for each voyage; and, because the Californian had not heard the distress signals of the Titanic, that ships maintain a 24-hour radio watch. The International Ice Patrol was established to warn ships of icebergs in the North Atlantic shipping lanes and to break up ice.

The Californian incident
The U.S. and British inquiries did little to end speculation and debate concerning the sinking of the Titanic. Particular focus centred on the Californian. Supporters of Lord, nicknamed “Lordites,” believed that the captain had been unfairly criticized. They held that a third ship—possibly the Samson, a Norwegian boat illegally hunting seals—was between the Leyland liner and the Titanic. That view eventually gained much support. Crew members of the Californian did not hear rockets being fired, though the sounds would have been audible if the ship had been within the distances claimed by U.S. and British investigators. In addition, people aboard the Titanic stated that a vessel was headed in their direction, which could not have been Californian, which was stopped at the time. While the true location of the Californian will likely never be conclusively known, many experts believe it was actually some 20 miles (37 km) away and would not have reached the Titanic before it sank. However, Lord has continued to draw criticism for his failure to take more action in response to the distress signals.

Discovery and legacy
bow of the Titanic, 2004
bow of the Titanic, 2004
Titanic Capt. Edward J. Smith's cabin, 2003
Titanic Capt. Edward J. Smith's cabin, 2003
Titanic: rusticles
Titanic: rusticles
Within days of the Titanic’s sinking, talk began of finding the wreck. Given the limits of technology, however, serious attempts were not undertaken until the second half of the 20th century. In August 1985 Robert Ballard led an American-French expedition from aboard the U.S. Navy research ship Knorr. The quest was partly a means for testing the Argo, a 16-foot (5-metre) submersible sled equipped with a remote-controlled camera that could transmit live images to a monitor. The submersible was sent some 13,000 feet (4,000 metres) to the floor of the Atlantic Ocean, sending video back to the Knorr. On September 1, 1985, the first underwater images of the Titanic were recorded as its giant boilers were discovered. Later video showed the ship lying upright in two pieces. While the bow was clearly recognizable, the stern section was severely damaged. Covering the wreckage were rust-coloured stalactite-like formations. Scientists later determined that the rusticles, as they were named, were created by iron-eating microorganisms, which are consuming the wreck. By 2019 there was a “shocking” level of deterioration, and a number of notable features, such as the captain’s bathtub, were gone.

ROV Hercules exploring Titanic wreck
ROV Hercules exploring Titanic wreck
The Titanic—located at about 41°43′57′′ N, 49°56′49′′ W (bow section), some 13 nautical miles (24 km) from the position given in its distress signals—was explored numerous times by manned and unmanned submersibles. The expeditions found no sign of the long gash previously thought to have been ripped in the ship’s hull by the iceberg. Scientists instead discovered that the collision’s impact had produced a series of thin gashes as well as brittle fracturing and separation of seams in the adjacent hull plates, thus allowing water to flood in and sink the ship. In subsequent years, marine salvagers raised small artifacts from the wreckage as well as pieces of the ship itself, including a large section of the hull. Examination of these parts—as well as paperwork in the builder’s archives—led to speculation that low-quality steel or weak rivets may have contributed to the Titanic’s sinking.

the filming of Titanic
the filming of Titanic
model ship used for the film Titanic
model ship used for the film Titanic
Branson, Missouri: Titanic Museum Attraction
Branson, Missouri: Titanic Museum Attraction
Titanic memorial in Washington, D.C.
Titanic memorial in Washington, D.C.
Countless renditions, interpretations, and analyses of the Titanic disaster transformed the ship into a cultural icon. In addition to being the subject of numerous books, the ship inspired various movies, notably A Night to Remember (1958) and James Cameron’s blockbuster Titanic (1997). In the late 20th and early 21st centuries, artifacts from the ship formed the basis of a highly successful exhibit that toured the world, and a profitable business was developed transporting tourists to the Titanic’s wreck. However, many opposed the removal of items, and the issue became highly contentious, complicated by the fact that the wreckage lies in international waters and is thus outside the jurisdiction of any country.


Several museums dedicated to the liner draw thousands of visitors each year; in 2012, the 100th anniversary of the ship’s sinking, Titanic Belfast opened on the site of Harland and Wolff’s former shipyard, and it became one of the city’s most popular tourist attractions. Although the wreck of the Titanic will eventually deteriorate, the famed liner seems unlikely to fade from the public imagination."""


import re

#Lets do some data pre-processing activities to minimize noises in our 
#training dataset that may cause training process difficult
def text_cleaner(text):
    # lowering all case
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # punctuations removal
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    long_words=[]
    # doesn't make sense to consider words that are shorter than 3 so remove those
    for i in newString.split():
        if len(i)>=3:                  
            long_words.append(i)
    return (" ".join(long_words)).strip()

# preprocess the data
data_new = text_cleaner(data_text)

#In this case we are training a lang model that will predict the next set of characetrs based on last 30 characetrs as a context.
#Below function will divide the entire dataset into sequnces of 30 characters
def create_seq(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
        # select sequence of tokens
        seq = text[i-length:i+1]
        # store
        sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    return sequences

# create sequences of 30 length 
sequences = create_seq(data_new)

#Now we enocde each character into a corresponding number using character mapping method
chars = sorted(list(set(data_new))) # all possible unqiue characters present in the given dataset
mapping = dict((c, i) for i, c in enumerate(chars))

def encode_seq(seq):
    sequences = list()
    for line in seq:
        encoded_seq = [mapping[char] for char in line]
        sequences.append(encoded_seq)
    return sequences

sequences = encode_seq(sequences)

#Once we are ready with our sequences, we split the data into training and validation splits. 
#This is because while training, I want to keep a track of how good my language model is doing on unseen data.
from sklearn.model_selection import train_test_split

# vocabulary size
vocab = len(mapping)
sequences = np.array(sequences)
# create X and y
X, y = sequences[:,:-1], sequences[:,-1]
# one hot encode y
y = to_categorical(y, num_classes=vocab)
# create train and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)

#Build the language model now
#I have used the embedding layer of Keras to learn a 20 dimension embedding for each character. This helps the model in understanding complex relationships between characters. 
#I have also used a GRU layer as the base model, which has 30 timesteps. Finally, a Dense layer is used with a softmax activation for prediction.

model = Sequential()
model.add(Embedding(vocab, 20, input_length=30, trainable=True))
model.add(GRU(30, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(vocab, activation='softmax'))
print(model.summary())

# compile the model
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
# fit the model
model.fit(X_tr, y_tr, epochs=100, verbose=2, validation_data=(X_val, y_val))

#Inference
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict character
		#yhat = model.predict(encoded, verbose=0)
   predict_x=model.predict(encoded) 
   yhat=np.argmax(predict_x,axis=1)

		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text

generate_seq(model, mapping, 19, "titanic was a very sad", 20)