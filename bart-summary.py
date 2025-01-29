from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""

text2 = """
The Israeli–Palestinian conflict is an ongoing military and political conflict about land and self-determination within the territory of the former Mandatory Palestine.[25][26][27] Key aspects of the conflict include the Israeli occupation of the West Bank and Gaza Strip, the status of Jerusalem, Israeli settlements, borders, security, water rights,[28] the permit regime, Palestinian freedom of movement,[29] and the Palestinian right of return.

The conflict has its origins in the rise of Zionism in the late 19th century in Europe, a movement which aimed to establish a Jewish state through the colonization of Palestine,[30][page needed][31] and the consequent first arrival of Jewish settlers to Ottoman Palestine in 1882.[32] The local Arab population increasingly began to oppose Zionism, primarily out of the fear of territorial displacement and dispossession.[32] The Zionist movement garnered the support of an imperial power in the 1917 Balfour Declaration issued by Britain, which promised to support the creation of a "Jewish homeland" in Palestine. Following British occupation of the formerly Ottoman region during World War I, Mandatory Palestine was established as a British mandate. Increasing Jewish immigration led to tensions between Jews and Arabs which grew into intercommunal conflict.[33][34] In 1936, an Arab revolt erupted demanding independence and an end to British support for Zionism, which was suppressed by the British.[35][36] Eventually tensions led to the UN adopting a partition plan in 1947, triggering a civil war.

During the ensuing 1948 Palestine war, more than half of the mandate's predominantly Palestinian Arab population fled or were expelled by Israeli forces. By the end of the war, Israel was established on most of the former mandate's territory, and the Gaza Strip and the West Bank were controlled by Egypt and Jordan respectively.[37][38] Since the 1967 Six Day War, Israel has been occupying the West Bank and the Gaza Strip, known collectively as the Palestinian territories. Two Palestinian uprisings against Israel and its occupation erupted in 1987 and 2000, the first and second intifadas respectively. Israel's occupation, which is now considered to be the longest military occupation in modern history, has seen it constructing illegal settlements there, creating a system of institutionalized discrimination against Palestinians under its occupation called Israeli apartheid. This discrimination includes Israel's denial of Palestinian refugees from their right of return and right to their lost properties. Israel has also drawn international condemnation for violating the human rights of the Palestinians.[39]

The international community, with the exception of the US and Israel, has been in consensus since the 1980s regarding a settlement of the conflict on the basis of a two-state solution along the 1967 borders and a just resolution for Palestinian refugees. The US and Israel have instead preferred bilateral negotiations rather than a resolution of the conflict on the basis of international law. In recent years, public support for a two-state solution has decreased, with Israeli policy reflecting an interest in maintaining the occupation rather than seeking a permanent resolution to the conflict. In 2007, Israel tightened its blockade of the Gaza Strip and made official its policy of isolating it from the West Bank. Since then, Israel has framed its relationship with Gaza in terms of the laws of war rather than in terms of its status as an occupying power. In a July 2024 ruling, the International Court of Justice rebuffed Israel's stance, determining that the Palestinian territories constitute one political unit and that Israel continues to illegally occupy the West Bank and Gaza Strip. The ICJ also determined that Israeli policies violate the International Convention on the Elimination of All Forms of Racial Discrimination. Since 2006, Hamas and Israel have fought several wars. Attacks by Hamas-led armed groups in October 2023 in Israel were followed by another war.[40] Israel's actions in Gaza since the start of the 2023 war have been described by international law experts, genocide scholars and human rights organizations as genocidal.[41]
"""
#print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

print(summarizer(text2, max_length=400, min_length=30, do_sample=False))
