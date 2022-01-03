# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


# used a dictionary to represent an intents JSON file

class ChatBot:
    def __init__(self):
        self.data = {"intents": [
            {"tag": "greeting",
             "patterns": ["Hello", "How are you", "How are you?", "Hi there", "Hi", "Whats up"],
             "responses": ["Howdy Partner!", "Hello", "How are you doing?", "Greetings!", "How do you do?"],
             },
            {"tag": "Store2Door",
             "patterns": ["What is Store2Door Cargo LLC", "What is Store2Door", "What do you do?", "What about you",
                          "What is store2door", "Store2Door", "store2door"],
             "responses": ["We are a freight company that caters to customers such as tourists, expats, students etc. "
                           "to deliver their personal cargo internationally and locally."
                           ],
             },
            {"tag": "Location",
             "patterns": ["Located", "located", "location", "Location"],
             "responses": ["We are based out of Dubai, United Arab Emirates."
                           ],
             },
            {"tag": "Timing",
             "patterns": ["your timings", "your timing", "timings please", "working hours", "working days",
                          "working days and hours"],
             "responses": ["We are open from Sundays through Thursdays from 09.00 am to 06.00 pm "
                           "and on Saturdays 09.00 am to 01.00 pm."
                           ],
             },
            {"tag": "use Store2Door",
             "patterns": ["use Store2Door", "use store2door", "use store to door"],
             "responses": ["Any individual or company who has a personal package to be sent "
                           "from the U.A.E to any part of the world or even within the U.A.E. can use Store2Door."
                           ],
             },

            {"tag": "age",
             "patterns": ["how old are you?", "when is your birthday?", "when was you born?"],
             "responses": ["I am 24 years old", "I was born in 1996", "My birthday is July 3rd and I was born in 1996",
                           "03/07/1996"]
             },
            {"tag": "date",
             "patterns": ["what are you doing this weekend?",
                          "do you want to hang out some time?", "what are your plans for this week"],
             "responses": ["I am available all week", "I don't have any plans", "I am not busy"]
             },
            {"tag": "name",
             "patterns": ["what's your name?", "what are you called?", "who are you?"],
             "responses": ["My name is Kippi", "I'm Kippi", "Kippi"]
             },
            {"tag": "goodbye",
             "patterns": ["bye", "g2g", "see ya", "adios", "cya"],
             "responses": ["It was nice speaking to you", "See you later", "Speak soon!"]
             },
            {"tag": "shipping",
             "patterns": ["What are the shipping modes you provide?", "What are the shipping mode?",
                          "What are the Delivery mode?",
                          "What are the consignment modes you provide?", "Whats the transporting mode?"],
             "responses": [
                 "We provide multiple modes of shipping option to the customer. Based on the urgency and cost effectiveness"
                 ", a customer can opt for any mode – sea, air, courier or land."]
             },
            {"tag": "quotation",
             "patterns": ["How can I get the quotation?", "How do you provide the quotation?",
                          "How can I get the tender?",
                          "How can I get the price?", "What will be the cost?",
                          "what are your estimate?", "Show me your estimate?", "What are the charges?"],
             "responses": [
                 "You can find live rates instantly on our website https://store2door.ae with all the options to the selected destination"
                 ". You can also send us an email for customized quote at info@store2door.ae."]
             },
            {"tag": "rates",
             "patterns": ["How soon can I get the rates?", "when i will recieve the rates?",
                          "When i will recieve the amount?"],
             "responses": ["The rates are available on our website and you can access it anytime anywhere."]
             },
            {"tag": "various locations",
             "patterns": ["Can you collect my packages from various locations?",
                          "Will you collect packages from various locations?"],
             "responses": [
                 "Yes, we can collect your packages from multiple locations. Be it stores, your home or your office"]
             },
            {"tag": "consolidate packages",
             "patterns": ["Do you consolidate packages picked up from various locations/stores?",
                          "Do you collect packages picked up from various locations/stores?",
                          "Do you consolidate packages picked up from various locations?",
                          "Do you consolidate packages picked up from various stores?", "Do you consolidate packages?",
                          "Consolidated packages",
                          "Can you Consolidated packages",
                          "Consolidation of packages"
                          ],
             "responses": ["Yes. We collect all the packages, consolidate and can send it as one shipment."]
             },
            {"tag": "freight charges",
             "patterns": ["Do you only provide the freight charges?", "Do you only provide the shipping charges?",
                          "Do you only provide the bulk charges?", "Do you only provide the payload charges?"],
             "responses": [
                 "We provide overall solution for all your shipping needs. We have “ADD ON” features like • Packing - To ensure your cargo can withstand the transit phase. "
                 "• Insurance - To cover uncertainties, unforeseen and uncontrollable situations. • Warehousing facility - To store your packages in our Dubai warehouse."]
             },
            {"tag": "restrictions on the size",
             "patterns": ["Are there restrictions on the size or the number of packages?",
                          "restrictions on the size or the number of packages", "restrictions of packasges",
                          "restictions on size of packages"],
             "responses": ["There is absolutely no restriction on the number of packages. "
                           "The size of the package allowed to ship would depend upon the mode selected and the carrier’s acceptance."]
             },
            {"tag": "store bags",
             "patterns": ["Can I store my bags/packages with you?", "store my bags", "store ny packages",
                          "store my packges with you", "store my bag with you"],
             "responses": ["We provide 7 calendar days free storage in Dubai for all your cargo. "
                           "The free time starts from the time the first item under a booking number is stored with us. "
                           "Thereafter, there will be charges based on the number of days per cbm."]
             },
            {"tag": "change shipping",
             "patterns": ["Can I change the mode of shipping after the order is collected?",
                          "change shipping mode when order is collected", "Change shipping mode"],
             "responses": ["We are the first company to offer multiple modes of shipping to individuals online."
                           " This allows us to give you the flexibility to choose your mode of shipment, destination, split your existing order as many times as you want to. "
                           "All you have to do is, call us at 04-4468444 during the business days or write to us at info@store2door.ae ."
                           " The rates will be revised accordingly. If you need any further assistance, please feel free to call us at 04-4468444 during the working hours."]
             },

            # Store Items
            {"tag": "store item",
             "patterns": ["Can I store any items in the warehouse?"],
             "responses": [
                 "Most general items can be stored. Items declared Dangerous goods by IATA will not be accepted for storage."]
             },
            {"tag": "store food item",
             "patterns": ["Can we store liquids and foods Items?", "Can we store liquids", "Can we store foods Items?"],
             "responses": [
                 "Since we do not accept Liquids and Food Items for shipping we do not accept it for storage."]
             },

            # Packaging
            {"tag": "prepare suitcase",
             "patterns": ["How should I prepare my suitcase or box for transport?", "Prepare suuitcase", "Prepare box",
                          "prepare suitcase for transport", "prepare suitcase for Box"],
             "responses": ["All goods should be packed in export worthy packaging (Cartons/Pallets/Crates), "
                           "which should be able to sustain the entire length of the travel. Assistance with packing can be provided, if required."]
             },
            {"tag": "goods specially packed",
             "patterns": ["Are the goods specially packed by Store2Door?", "goods specially packed by Store2door",
                          "Are good packed well by store2door?", "Is good packed very well by store2door?"],
             "responses": [
                 "If the initial packing of the items is packed appropriately to withstand the transit, then we do not recommend any additional packing. However, upon receiving it in the warehouse, "
                 "if we determine the packaging is not air or sea worthy then we intimate you. The charges are available on our website."]
             },
            {"tag": "types of packaging",
             "patterns": ["What are the types of packaging you offer?",
                          "what are the different types of packaging you offer",
                          "How many types packaging you provide"],
             "responses": [
                 "We do all kinds of packing right from cartons to pallet to crate for regular and fragile items"]
             },
            {"tag": "additional packaging",
             "patterns": ["What if we don’t want to pay for the additional packing and do it ourselves?",
                          "Can we do additional packaging by ourselves"],
             "responses": ["We have a professional team who knows the kind of packing particular items require. "
                           "If you do not wish to pay additional charges for packing and the carrier accepts the items in its original packing, then we will ship out the package. "
                           "However, if the packing is found unworthy for sea or air freight mode then it will not be accepted by the carrier till it is appropriately packed. "
                           "We will inform you and after you approve and pay the additional charges for the packing, we will proceed further."]
             },
            {"tag": "damage of goods",
             "patterns": ["Who takes the responsibility for the damage in case the goods are not packed properly?",
                          "responsible for damage of goods", "Goods damaged while packaging"],
             "responses": [
                 "It is the customer's responsibility to pack the goods in export worthy packages. You also need to "
                 "take care of internal packing. "
                 "We do offer professional packing as an “Add on” service. You can opt for it on our website by paying the additional charges."]
             },
            {"tag": "Internal packing",
             "patterns": ["What is internal packing?", "internal packing"],
             "responses": [
                 "Internal packing is creating a cushion for the items to remain safe inside the package during the journey."
                 "Even with sturdy packing from outside items can get damaged internally if the internal packing is weak."]
             },
            {"tag": "package opened",
             "patterns": ["Will my packages be opened by anyone?", "will anyone open my package",
                          "Can anyone open the package",
                          "Who can open my package"],
             "responses": [
                 "Yes, we reserve the right to open the packages to check the contents, if we find it necessary. "
                 "However, the packages could be opened by Customs Officer either at origin or destination or both, if they want to inspect."]
             },
            {"tag": "goods handle",
             "patterns": ["Do I have to indicate specifically, in case I want the goods to be handled with care?",
                          "how do you care fragile items", "how do you care delicate items"],
             "responses": [
                 "Yes. Our team should be notified about any delicate/fragile items. Any fragile items should be packed with proper care."]
             },
            {"tag": "damaged good",
             "patterns": [
                 "If the packing was done by Store2Door and the package is found damaged due to which the items were damaged or stolen, will you be responsible for it?",
                 "Who is responsible for damaged goods?", "Who is responsible for Stolen goods?",
                 "who is responsible for damaged goods at delivery time?"],
             "responses": ["We do professional packing which is good to withstand the journey. "
                           "However, a package can be damaged or item can be stolen enroute the journey as it moves through multiple hands. Therefore, we recommend you do cargo insurance for any eventualities. "
                           "We will not be responsible for any damage or stolen items.."]
             },

            # Collection
            {"tag": "address collect",
             "patterns": ["What kind of addresses can you collect from and deliver to?",
                          "from where can you collect the packages?",
                          "what diffrenet types of address where you collect and deliver?"],
             "responses": [
                 "We can collect from stores, warehouses, hotels, hostels, residences etc. and deliver the same to these places at the destination."]
             },
            {"tag": "pacakges location",
             "patterns": ["Can you collect my packages from various locations?", "various location",
                          "packages from various location"],
             "responses": [
                 "Yes, we can collect your packages from multiple locations. Be it stores, your home or your office."]},
            {"tag": "consolidated pakages",
             "patterns": ["Do you consolidate packages picked up from various locations/stores?"],
             "responses": ["Yes. We collect all the packages, consolidate and can send it as one shipment."]},
            {"tag": "collection time",
             "patterns": ["What time my collection will take place?", "How much time my collection will take place?",
                          "Pickup time "],
             "responses": [
                 "Our customer service representative will contact you to confirm the pick time and location."]},
            {"tag": "call text",
             "patterns": ["Will I receive a call/text before my collection?",
                          "will recieve any text before collection?",
                          "will get any call before collection?",
                          "Will get any call or messages before collecting packages?"],
             "responses": ["Yes. The contact person given for the pick up will be called before the collection."]},
            {"tag": "change collection date",
             "patterns": ["Can I change my collection date?", "can i change date for collecting packages?",
                          "change collection date packages?"],
             "responses": [
                 "Yes. We must be intimated about the change by mail or phone before the collection is arranged."]},
            {"tag": "personal details",
             "patterns": ["Can I change my collection name, collection address or telephone number?",
                          "Can i change mobile number?", "can i change my name?",
                          "can i change my collection address?"],
             "responses": ["Yes. You may change the collection address before the collection is arranged."]},
            {"tag": "missed pacakge",
             "patterns": ["What if I miss the pick up?", "missed the pick up package?"],
             "responses": ["You will need to call us again to rebook the pickup and it may incur additional charges."]},
            {"tag": "university rooms",
             "patterns": ["Do you collect from university dorm rooms?", "Do you collect from University?",
                          "Do you collect pacakges from University?", "Can you collect from University?"],
             "responses": ["Yes. We do pick up from the university dorm rooms"]},
            {"tag": "weekends",
             "patterns": ["Do you collect on weekends or Bank Holidays?", "Do you collect on weekends?",
                          "Do you collect Bank Holidays?", "Do you collect on National Holidays?"],
             "responses": [
                 "We do not collect on the weekends and on Bank Holidays; however, we can make an exception on case to case basis. Additional charges may apply"]},
            {"tag": "time slots",
             "patterns": ["Are collection time slots available?", "What are time slots available?",
                          "time slots availability",
                          "time slots?"],
             "responses": ["Usually the collection times will be between 9 am to 6 pm (Saturdays through Thursdays)"]},
            {"tag": "same day collection",
             "patterns": ["Do you offer same day collections?", "same day collection?"],
             "responses": [
                 "Yes. We can arrange same day collection on a working day; however, we need to be informed before 4:00 pm same day. "
                 "There will be additional charges for express/same day pick up."]},
            {"tag": "update time slots",
             "patterns": ["Can I update my collection time slot?", "update collection time slots?",
                          "update time slots?"],
             "responses": ["Yes. It should be done a day before collection."]},
            {"tag": "additional information",
             "patterns": [
                 "My address is difficult to find and is not very accessible, can I provide additional access information?",
                 "can i provide additional information?",
                 "address is difficult, add another address"],
             "responses": ["Yes. You may call our helpline and provide the details."]},
            {"tag": "bussiness address",
             "patterns": ["Can you collect from a business address?",
                          "Can you collect package from a business address?"],
             "responses": ["Yes. We can collect from a business address."]},
            {"tag": "drop package",
             "patterns": ["Can I drop my bag off?", "Can I drop pacakge at your warehosue?"],
             "responses": ["Yes. You can drop off the packages at our warehouse."]},
            {"tag": "person availability",
             "patterns": ["Do I need to be present when my bag is collected?",
                          "My present is compulsory while collecting pacakges?"],
             "responses": ["No. It is not required for you to be present when your bag is collected."]},
            {"tag": "person unavailability",
             "patterns": ["What if I need to go out on the collection day?",
                          "what if i am not present on collection day?",
                          "what if i am not available on collection day?"],
             "responses": [
                 "You may change the date of collection a day before or leave the package with someone for us to collect."]
             },
            # PAYMENTS
            {"tag": "payment method",
             "patterns": ["What payment methods do you accept?", "What are the payment method do you accept?",
                          "payment method"],
             "responses": ["You can make the payment online on our website. "
                           "We accept debit card, credit card and cash as well. You may also transfer the payment directly to our bank account or deposit cash into our account."]},
            {"tag": "outstanding payment",
             "patterns": ["How do I make an outstanding payment?", "outstanding payment?",
                          "how can i make an outstanding payment?"],
             "responses": [
                 "You can make the payment online on our website by a credit card or a debit card. We accept all cards. We can also accept payment in cash."]},
            {"tag": "pay shiping",
             "patterns": ["When do I have to pay for the shipping?", "When to pay for shipping?", "pay for shiping"],
             "responses": [
                 "You can pay at the time of placing the booking which will be considered as an advance payment. "
                 "If you haven’t made the advance payment, then we need you to make the payment in full once our invoice is raised."]},
            {"tag": "credit card",
             "patterns": ["Do you hold credit card details?", "Do you accept credit card",
                          " do you like payment through credit card?"],
             "responses": ["No. We do not hold any credit card details."]},
            {"tag": "card declined",
             "patterns": ["Why has my card been declined?", "why card declined", "card declined"],
             "responses": ["You will need to check with your bank or your service provider."]},
            {"tag": "cash",
             "patterns": ["Can I pay in cash at the time of delivery?", "cash on delivery?", "cod"],
             "responses": [
                 "To safeguard ourselves, as a company policy, we collect the entire payment as soon as the booking is confirmed and then ship your packages."]},
            {"tag": "cheque or cash",
             "patterns": ["Can I pay by cheque or cash?", "do you accept cash or cheque"],
             "responses": ["Yes, you may pay by current dated cheque or cash in full. "
                           "In case of cheque payments, the shipment will be executed only after the cheque has been encashed."]},
            {"tag": "installment",
             "patterns": ["Can I make part payments/installments?", "can i pay in installement",
                          "Can i make payment in parts?"],
             "responses": [
                 "You can make the part payments or advance payments; however, the shipment will be processed and shipped out after the complete payment is received."]},
            {"tag": "currency",
             "patterns": ["In what currency do you accept the payments?", "in which currency do you accept payments"],
             "responses": ["We accept payments only in Dirhams (AED)."]
             },

            # INSURANCE AND CLAIMS
            {"tag": "insurance premium",
             "patterns": ["How much is the insurance premium?", "insurance premium?"],
             "responses": [
                 "The acceptance of insurance proposal & Insurance Premium will depend on the value & nature of goods. "
                 "You may check the charges by going on to the website and entering the value of the Goods."]
             },
            {"tag": "insured package",
             "patterns": ["How much should the package be insured for?", "what shoul be the insured package?"],
             "responses": ["The package should be insured for the present value of the goods + the freight amt."]
             },
            {"tag": "full value damaged",
             "patterns": ["Do I get the full value back for the damaged goods?",
                          "do i get full refund if pacakge is damaged?"],
             "responses": [
                 "Yes. After the assessment by the insurance company that the value of the damaged goods declared by the customer is correct and not overvalued the full claim amount is settled deducting only the Excess Fee which is currently Aed 500/- for processing the claim."]
             },
            {"tag": "calculate insurance",
             "patterns": ["How do you calculate the insurance premium fee?", "insurance premium fee",
                          "How you calculate insurance premium fee?"],
             "responses": [
                 "A certain percentage is considered based on the invoice value of the goods and the freight cost. "
                 "Kindly visit our website to know the exact premium for your cargo."]
             },
            {"tag": "mandatory insurance",
             "patterns": ["Is it mandatory to take insurance?", "it is compulsory to take insurance?"],
             "responses": [
                 "Insurance is not mandatory; however, we strongly recommend insuring your goods to have coverage in case of eventualities."]
             },
            {"tag": "coverage",
             "patterns": ["What is the coverage?", "coverage"],
             "responses": ["Loss/Damage to your goods as per the policy Terms & Conditions."]
             },
            {"tag": "later insurance",
             "patterns": ["Can I add the insurance later?", "can insurance be added later"],
             "responses": ["It is ideal to incept coverage prior to commencement of shipment from Store/Origin. "
                           "However, you can opt for it later but the coverage will be from the commencement date of the policy."]
             },
            {"tag": "clim insurance",
             "patterns": ["How do I claim for the insurance in case of damage, missing or stolen items?",
                          "how to claim insurance?",
                          "How to claim insurance in case of damage, missing or stolen items?"],
             "responses": [
                 "You can write to us at info@store2door.ae to register the claim with insurance company with your booking number in the subject and from thereon, we will guide you."]
             },
            {"tag": "insurance process",
             "patterns": ["How many days does it take for the insurance to process?", "insurance procedure"],
             "responses": [
                 "It takes one day to open the claim file. There is no definite time-frame given by the insurance company in which they can complete the investigation. "
                 "However, once the claim is approved, it takes upto 10-15 days to get the insurance claim amount subject to submission of required documents."]
             },
            {"tag": "replace damaged",
             "patterns": ["Can insurance replace the damaged item for a new item?",
                          "Can damaged item be replace ny new item? "],
             "responses": [
                 "Policy will indemnify for your loss/damage. If the damaged item is at beyond repair conditions and/or not economical to repair, "
                 "insurance company as per the assessment shall declare it as ‘Total Loss’ at their own discretion and settle the claim."],
             },
            {"tag": "claim money",
             "patterns": ["Do I get the claim money in the account or in cheque?",
                          "claim money in the account or in cheque"],
             "responses": ["It can be a cheque or an online transfer."]
             },
            {"tag": "claim form",
             "patterns": ["Where can I find the claim processing form?", "Who will provide claim processing form?",
                          "Where to insurance claiming process"],
             "responses": ["Store2Door will provide the claim processing form upon request."]
             },
            {"tag": "insurance company",
             "patterns": ["What is the name of the insurance company?", "name of the insurance company?",
                          "name of the insurance organization?"],
             "responses": [
                 "The current insurance company is AXA Insurance Gulf BSC. We will keep you informed in case the insurance company changes in the future."],
             },
            {"tag": "insurance items",
             "patterns": ["Is the insurance only for new items or also for used items?",
                          "Is insurance valid for new items?"],
             "responses": [
                 "The insurance can be done for both old and new items. The premium; however, will be higher for the old/used items."]
             },
            {"tag": "insurance validation",
             "patterns": ["Till when is the insurance valid?", "insurance validation", "insurance validity"],
             "responses": [
                 "The insurance is only valid from the date of issue of policy till it reaches the port of destination/door delivery requested. In door delivery cases, if the shipment has reached the destination port and not collected within 30 days, the policy seizes to exist. "
                 "Where the customer wants the services only up to the port of destination, the insurance seizes to exist once it reaches the final port."]
             },
            {"tag": "claim damaged",
             "patterns": ["My contents have been damaged, how do I claim for the damage?", "claim for the damage goods",
                          "How to claim insurance when package is damaged?"],
             "responses": [
                 "When you receive the goods in damaged condition please put a remark on the delivery note that the shipment has been received in damaged condition while receiving the package and take a picture as proof. "
                 "You may then contact us and we will guide you through the process of claim."]
             },
            {"tag": "insurance transfer",
             "patterns": ["Is the insurance transferrable?", "can we transfer insurance"],
             "responses": ["No. The insurance is not transferrable."]
             },
            {"tag": "cancel insurance",
             "patterns": ["Can I cancel the insurance?", "cancel insurance", "How to cancel insurance"],
             "responses": ["If the insurance policy has been issued, it CANNOT be cancelled."]
             },

            # QUOTATIONS
            {"tag": "sea mode",
             "patterns": ["Is Sea mode the cheapest mode of shipping?", "Sea mode of shipping?"],
             "responses": ["No. For smaller sized items Courier is the fastest and the cheapest option. "
                           "The next best is Airfreight which are still cheaper than Sea freight for slightly bigger shipments. But Sea frt mode is best for large shipment. "
                           "But to check which is the fastest or cheapest mode check the rates on Store2door App/Website as it will vary from country to country."]
             },
            {"tag": "rates negotiable",
             "patterns": ["Can the rates mentioned on the website be negotiated?", "is the rates negotiable"],
             "responses": ["Our rates are very economical and therefore fixed."]
             },
            {"tag": "save quotation",
             "patterns": ["Can I save the quotation and lock the pricing?",
                          "is the quotation and lock pricing can be saved?"],
             "responses": [
                 "You can save the quotation and retrieve it from “My Quotations” section on our website by logging in to your account. "
                 "However, it is imperative for you to “Re-validate” your quote when you decide to place the booking. "
                 "In general, the quote is valid for 7 days from the date of initial quotation, if no changes have been made to the cargo details, destination or/and mode of shipment."]
             },
            {"tag": "rate at door",
             "patterns": ["I can’t see the rates up to the door delivery?", "rates up to door delivery?"],
             "responses": [
                 "If you don’t see the rates upto the door then you may write to us at info@store2door.ae or click on “Customized quote request” on the website."]
             },
            {"tag": "door delivery rates",
             "patterns": ["I want door delivery but find the rates very high?"],
             "responses": [
                 "We have very competitive rates; however, if you find the rates too high for door to door delivery, you can opt for shipping from door to port. "
                 "At destination, you can do the clearance yourself or appoint a clearing agent on your behalf."]
             },
            {"tag": "quotation currencey",
             "patterns": ["Can I get the quotation in my country’s currency?",
                          "what the quotation of country currency?"],
             "responses": [
                 "We can manually provide the rates in the requested currency of your country; however, the payment must be settled in Dirhams (AED) only."]
             },
            {"tag": "quotation other services",
             "patterns": ["Besides freight charges, do you also provide quotations for other services?",
                          "Do you provide quotation for other services"],
             "responses": [
                 "Yes, besides freight we can quote you for our “Add-On” services such as warehousing, insurance and packing."]
             },

            # DOCUMENTS REQUIRED
            {"tag": "documents required",
             "patterns": ["What are the documents required?", "docments required"],
             "responses": [
                 "The standard documents required are a clear scanned copy of your passport, visa page, Emirates ID (U.A.E. nationals & U.A.E. residents only). "
                 "The mandatory documents will depend upon destination country rules and regulations."]
             },
            {"tag": "documents related",
             "patterns": ["Do you need any documents related to the goods?",
                          "do you require documents related to goods?"],
             "responses": [
                 "We need the copy of the purchase invoice in case of the new items. For household items, a detailed packing list of the items is required."]
             },
            {"tag": "passport copy",
             "patterns": ["Do I need to provide my passport copy?", "passport copy", "is passport copy is required"],
             "responses": [
                 "The customs need to relate the goods to you and therefore, a copy of the passport is required as proof."]
             },
            {"tag": "send package",
             "patterns": ["I do not have the purchase invoice/bill. Can I still send the packages through you?",
                          "How can i send the packages through you?"],
             "responses": [
                 "Please provide us the value of the item and we will prepare the shipping documents accordingly."]
             },
            {"tag": "submit documents",
             "patterns": ["How do I submit the documents to you?", "submit documents", "how can i submit documents"],
             "responses": [
                 "You can simply log into your account with Store2Door and upload the required documents from “My Documents” section."]
             },
            {"tag": "shipping document",
             "patterns": [
                 "Where can I find my shipping documents (Insurance policy copy, bill of lading, airway bill etc.)?",
                 "where to find shipping documents"],
             "responses": [
                 "You can view and download the shipping documents from “My Documents” section of your Store2Door account."]
             },

            # TRACKING
            {"tag": "track order",
             "patterns": ["How do I track my order?", "track order", "how to track order"],
             "responses": [
                 "You can track the progress of your shipment from our website https://store2door.ae with the booking reference number."]
             },
            {"tag": "shipping status",
             "patterns": ["It’s been more than 24 hours and the status of the shipment hasn’t changed?",
                          "shipping status hasn't changed"],
             "responses": [
                 "Please check your emails for communication from us as all the updates will be sent automatically to your mailing address."]
             },
            {"tag": "unable to track",
             "patterns": ["The website is down and I am unable to track the order?", "Unable to track the order"],
             "responses": ["In such instances when our website is down for some upgrade or maintenance purposes, "
                           "you can send us an email at info@store2door.ae with your booking reference number in the subject line and we will revert to you with the status."]
             },
            {"tag": "hold shipping",
             "patterns": [
                 "The tracking shows that my order has been shipped, I want to add a few more items, can you please hold the shipment back?"],
             "responses": [
                 "Once the order has be shipped out, there is no way we can hold it back or modify any of the details. "
                 "You can simply place a new order with us in case you want to ship more items."]
             },

            # CUSTOMS
            {"tag": "custom effect",
             "patterns": ["Do customs affect me?", "How custom effect me?", "custom effect"],
             "responses": ["Usually all new items attract duty as per the tariffs in their respective countries."
                           " But some countries are more lenient than others and may allow some cargo to be cleared without payment of duty which will be completely at the discretion of the Customs Department"]
             },
            {"tag": "custom duty",
             "patterns": ["Do I need to pay Customs Duty and Taxes?", "is custom duty compulsory", "custom duty"],
             "responses": ["Yes. You need to pay Customs Duty and Taxes at the destination. "
                           "Usually old and used personal effects are exempted from Duty or a duty discount is given depending upon the country of import, but it will be at the discretion of the customs. "
                           "New Items will attract customs duty and taxes."]
             },
            {"tag": "information clearance",
             "patterns": ["What Information do I need to provide to achieve duty free clearance?",
                          "what information should i provide for duty free"],
             "responses": ["Usually all new items attract duty as per the tariffs in their respective countries. "
                           "Only household goods that are considered as old and used are exempted from duty."]
             },
            {"tag": "declared value",
             "patterns": ["What is the 'Declared Value for Customs?", "declared value"],
             "responses": [
                 "Declared Value for Customs is the correct and fair value of the goods to be declared to customs."]
             },
            {"tag": "pay custom",
             "patterns": ["How much customs duty will I need to pay?", "How much amount to pay for custom"],
             "responses": ["Customs Duty will be as per the import tariffs of their respective countries."]
             },
            {"tag": "cleareance destination",
             "patterns": ["How long will custom clearance take at destination?", "cleareance destination"],
             "responses": [
                 "Usually custom clearance at destination takes 2-3 days. However, it can take more than usual if the customs need additional information or clarification."]
             },
            {"tag": "commercial items",
             "patterns": ["Can I send commercial items?", "How can i send commercial items"],
             "responses": ["No. Individuals are not allowed to send commercial items."]
             },
            {"tag": " shipment custom",
             "patterns": ["Why is my shipment still in customs?" "is shipment still in custom"],
             "responses": [
                 "If the package has been held back at customs, it could mean customs have some reservations on the value of the goods or the goods itself."]
             },
            {"tag": "custom contact",
             "patterns": ["Customs have contacted me directly. What do I do now?", "what to do wHen custom contact me"],
             "responses": ["Mostly customs might call you if they need some clarification. "
                           "If they are satisfied with your reply they will release the shipment or else they will call you to their office for further clarification."]
             },
            {"tag": "custom bill",
             "patterns": ["I have received a bill from customs, how do I dispute this?",
                          "what to do when i receive bill from custom"],
             "responses": [
                 "You will have to approach the customs office and explain to the customs officer with the proof of purchase that the declared value and item are correct."]
             },
            {"tag": "upload document",
             "patterns": ["Why do I need to upload a copy of my flight ticket / passport / work permit /visas?",
                          "is compulsory to upload documents"],
             "responses": [
                 "This is to identify that you are a legal person staying in the destination country and to relate the packages belong to you."]
             },
            {"tag": "unavilable document",
             "patterns": ["What if I am unable to provide a flight ticket, passport, visa etc.?",
                          "unavailable to provide ticket/passport/vissa"],
             "responses": [
                 "If any of the standard and mandatory documents are not provided, then the shipment cannot be shipped."]
             },
            {"tag": "custom paperwork",
             "patterns": ["Whose name needs to appear on the customs paperwork?",
                          " wHat name to be appear on custom paperwork"],
             "responses": ["The receiver’s name will appear on the custom’s paperwork"]
             },
            {"tag": "shipment delayed",
             "patterns": ["Can the shipment be delayed at the customs?", "why shipment is delayed"],
             "responses": [
                 "Yes. If the customs officer has some doubts or suspects something wrong, he can hold the shipment for inspection."]
             },
            {"tag": "storage charges",
             "patterns": ["Will I have to pay storage charges due to customs delay?", "storage cHarges"],
             "responses": [
                 "Yes, you will be liable to pay any charges including storage, packing/re-packing charges due to customs inspection."]
             },

            # PROHIBITED ITEMS
            {"tag": "items restricted",
             "patterns": ["What can I send? Is there anything I can't send?", "what items are restricted?"],
             "responses": [
                 "Store2Door does not send Dangerous goods/ Perishable Goods / Food Items / Currency/Jewelry, prohibited items at destination country etc. "
                 "Please refer to the General Prohibited List on our website or App."]
             },
            {"tag": "liquids",
             "patterns": ["Can I send toiletries and other liquids?", "toiletries an other liquids"],
             "responses": ["Store2Door does not carry toiletries and liquids."]
             },
            {"tag": "plastics items",
             "patterns": ["Can I send laundry bags, bin liners or plastic storage boxes?",
                          "can i send plastics bags, liners, laundary bags"],
             "responses": [
                 "Yes. You may send the above-mentioned items if these are packed in export worthy packing (Cartons/Pallets/Crates) etc."]
             },
            {"tag": "electronic items",
             "patterns": ["Can I send a laptop, TV, phone, printer or other electrical items?",
                          "Can i send electronic items"],
             "responses": [
                 "Yes. One battery per unit, ideally fitted into the unit is allowed. Loose batteries are not allowed."]
             },
            {"tag": "sports items",
             "patterns": ["Can I send sports equipment such as a bicycle, skis or golf clubs?",
                          "can i send sports items"],
             "responses": [
                 "Yes. If the sports equipment are packed in export worthy packing that is acceptable by the carrier."]
             },
            {"tag": "food items",
             "patterns": ["Can I send food?", "can you send food items"],
             "responses": ["No. We do not carry food items or any perishable items."]
             },
            {"tag": "musical instruments",
             "patterns": ["Can I send musical instruments such as Guitars?", "can you send musical instruments?"],
             "responses": [
                 "Yes. If the musical instruments are packed in export worthy packing that is acceptable by the carrier."]
             },
            {"tag": "batteries",
             "patterns": [" Can I ship batteries?", "can yuu ship battery"],
             "responses": ["No. Only batteries are not allowed."]
             },

            # # DELIVERY
            {"tag": "address deliver",
             "patterns": ["What kind of addresses can you collect from and deliver to?",
                          "from where you collect and where can you deliver?"],
             "responses": [
                 "We can collect from most places in the U.A.E. and deliver to most parts of the world. The embargo countries/war inflicted countries are avoided."]
             },
            {"tag": "delivery status",
             "patterns": ["How do I know when my bag will be out for delivery?", "delivery status",
                          "when my bag will be out for delivery?"],
             "responses": ["Our team at the destination will be calling you before arranging the delivery."]
             },
            {"tag": "change data",
             "patterns": ["Can I change my delivery name, delivery address or telephone number?",
                          "can i chage my data?"],
             "responses": [
                 "You may do it before delivery of the shipment so that any difference in billing can be paid before dispatch."]
             },
            {"tag": "package arrival",
             "patterns": ["How long it will take for my package to arrive?",
                          "How much tie it will take for my package to arrive?"],
             "responses": [
                 "The approximate transit time for each service is mentioned in your quotation. The transit times will be subject to no delay in customs and availability of space on the carriers. The delay could happen due to force majeure as well as natural or man-made calamities"]
             },
            {"tag": "package delivered time",
             "patterns": ["What time will my package be delivered? Will I receive a telephone call?", ],
             "responses": ["Yes, you will receive a call before delivery of the package."]
             },
            {"tag": "delivery missed",
             "patterns": ["What do I do if I miss the delivery driver?", "what to do if i missed the deilvery"],
             "responses": [
                 "Please call back again to re-arrange the delivery. This may attract additional transportation charges"]
             },
            {"tag": "failed delivery",
             "patterns": ["Can I re-arrange a failed delivery?", "what to do if i re arrange a failed delivery"],
             "responses": [
                 "Please call back again to re-arrange the delivery. This may attract additional transportation charges."]
             },
            {"tag": "estimated delivery",
             "patterns": ["It is my estimated delivery date and my package hasn't been delivered?",
                          "what to do estimated delivery hasnt been delivered"],
             "responses": [
                 "You may track the status of you booking on our website or you may write to us at info@store2door.ae"]
             },
            {"tag": "sign pacakage",
             "patterns": ["Can someone else sign for my package? Do I have to be there?",
                          "can others also sign my packge?"],
             "responses": [
                 "We need an authorization email at infor@store2door.ae instructing us to hand over the packages to another person."]
             },
            {"tag": "change date",
             "patterns": ["Can I change my delivery date?", "Change delivery date"],
             "responses": [
                 "Yes. You may change the delivery date but there could be storage charges depending on the length of the delay."]
             },
            {"tag": "cargo",
             "patterns": ["Can I collect my cargo?", "cllect cargo"],
             "responses": ["Yes. You may come to our warehouse and collect the cargo."]
             },
            {"tag": "items not delivered",
             "patterns": ["Why haven’t all my items been delivered?", "why all items not bee delivered?"],
             "responses": [
                 "If all the items are not delivered, intimate us immediately at info@store2door.ae and we will investigate why all the items were not delivered."]
             },
            {"tag": "late delivery",
             "patterns": ["Can I claim for late delivery?", "claim for late delivery"],
             "responses": ["No. Usually deliveries are on time unless delay due to customs."]
             },
            {"tag": "package not recived",
             "patterns": ["My bag is showing as delivered but I haven’t received it?",
                          "its showing delivered bt not recieved it"],
             "responses": ["Please send us an e mail to cross check the status of shipment."]
             },
            {"tag": "delivery missing",
             "patterns": ["I keep missing my delivery, what can I do?", "delivery is missing"],
             "responses": [
                 "Please request someone else to receive the delivery on your behalf with an email to us authorizing to deliver or you may come and collect your package from our warehouse at your convenience within the warehouse timings."]
             },
            {"tag": "storage pacakge",
             "patterns": ["Can you hold my package? Do you offer a storage service?"],
             "responses": ["Yes. We can hold your package for long term storage. Please contact us for charges."]
             },
            {"tag": "driver",
             "patterns": ["What if the driver cannot access my address?", "Driver cant access my address",
                          "driver doesnt know my address"],
             "responses": ["You may provide an alternative address or come and collect the package from our warehouse."]
             },
            {"tag": "redirect address",
             "patterns": ["Can I redirect to a different delivery address?", "Can i give another delivery address?"],
             "responses": [
                 "Yes. You may change the final delivery address. If the distance is more than the earlier distance, additional charges will be applicable."]
             },
            # REFUNDS & CANCELATIONS
            #     {}

        ]}

        self.words = []
        self.classes = []
        self.doc_X = []
        self.doc_Y = []
        self.train_X = None
        self.train_Y = None
        self.lemmatizer = WordNetLemmatizer()
        self.model = None

    def download_data(self):
        nltk.download("punkt")
        nltk.download("wordnet")

    def lemmatize_data(self):
        for intent in self.data["intents"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                self.words.extend(tokens)
                self.doc_X.append(pattern)
                self.doc_Y.append(intent["tag"])

            if intent["tag"] not in self.classes:
                self.classes.append(intent["tag"])
        words_loc = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in string.punctuation]
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))

    def list_training_data(self):
        training = []
        out_empty = [0] * len(self.classes)

        for idx, doc in enumerate(self.doc_X):
            bow = []
            text = self.lemmatizer.lemmatize(doc.lower())
            for word in self.words:
                bow.append(1) if word in text else bow.append(0)
            output_row = list(out_empty)
            output_row[self.classes.index(self.doc_Y[idx])] = 1
            training.append([bow, output_row])
        random.shuffle(training)
        training = np.array(training, dtype=object)
        self.train_X = np.array(list(training[:, 0]))
        self.train_Y = np.array(list(training[:, 1]))

    def train_data(self):
        input_shape = (len(self.train_X[0]),)
        output_shape = len(self.train_Y[0])
        epochs = 200
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=input_shape, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(output_shape, activation="softmax"))
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=["accuracy"])
        print(self.model.summary())
        self.model.fit(x=self.train_X, y=self.train_Y, epochs=300, verbose=1)

    def clean_text(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def bag_of_words(self, text, vocab):
        tokens = self.clean_text(text)
        bow = [0] * len(vocab)
        for w in tokens:
            for idx, word in enumerate(vocab):
                if word == w:
                    bow[idx] = 1
        return np.array(bow)

    def pred_class(self, text, vocab, labels):
        bow = self.bag_of_words(text, vocab)
        result_pred = self.model.predict(np.array([bow]))[0]
        thresh = 0.2
        y_pred = [[idx, res] for idx, res in enumerate(result_pred) if res > thresh]

        y_pred.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in y_pred:
            return_list.append(labels[r[0]])
        return return_list

    def get_response(self, intents_list, intents_json):
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        result_resp = 'Not understand'
        for i in list_of_intents:
            if i["tag"] == tag:
                result_resp = random.choice(i["responses"])
                break
        return result_resp

