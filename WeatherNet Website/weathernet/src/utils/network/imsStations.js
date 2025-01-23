const imsStations = [
  {
    id: "1",
    label: "Jerusalem",
  },
  {
    id: "2",
    label: "Tel Aviv Coast",
  },
  {
    id: "3",
    label: "Haifa",
  },
  {
    id: "4",
    label: "Rishon le Zion",
  },
  {
    id: "5",
    label: "Petah Tiqva",
  },
  {
    id: "6",
    label: "Ashdod",
  },
  {
    id: "7",
    label: "Netania",
  },
  {
    id: "8",
    label: "Beer Sheva",
  },
  {
    id: "9",
    label: "Bnei Brak",
  },
  {
    id: "10",
    label: "Holon",
  },
  {
    id: "11",
    label: "Ramat Gan",
  },
  {
    id: "12",
    label: "Asheqelon",
  },
  {
    id: "13",
    label: "Rehovot",
  },
  {
    id: "14",
    label: "Bat Yam",
  },
  {
    id: "15",
    label: "Bet Shemesh",
  },
  {
    id: "16",
    label: "Kfar Sava",
  },
  {
    id: "17",
    label: "Herzliya",
  },
  {
    id: "18",
    label: "Hadera",
  },
  {
    id: "19",
    label: "Modiin",
  },
  {
    id: "20",
    label: "Ramla",
  },
  {
    id: "21",
    label: "Raanana",
  },
  {
    id: "22",
    label: "Modiin Illit",
  },
  {
    id: "23",
    label: "Rahat",
  },
  {
    id: "24",
    label: "Hod Hasharon",
  },
  {
    id: "25",
    label: "Givatayim",
  },
  {
    id: "26",
    label: "Kiryat Ata",
  },
  {
    id: "27",
    label: "Nahariya",
  },
  {
    id: "28",
    label: "Beitar Illit",
  },
  {
    id: "29",
    label: "Um al-Fahm",
  },
  {
    id: "30",
    label: "Kiryat Gat",
  },
  {
    id: "31",
    label: "Eilat",
  },
  {
    id: "32",
    label: "Rosh Haayin",
  },
  {
    id: "33",
    label: "Afula",
  },
  {
    id: "34",
    label: "Nes-Ziona",
  },
  {
    id: "35",
    label: "Akko",
  },
  {
    id: "36",
    label: "Elad",
  },
  {
    id: "37",
    label: "Ramat Hasharon",
  },
  {
    id: "38",
    label: "Karmiel",
  },
  {
    id: "39",
    label: "Yavneh",
  },
  {
    id: "40",
    label: "Tiberias",
  },
  {
    id: "41",
    label: "Tayibe",
  },
  {
    id: "42",
    label: "Kiryat Motzkin",
  },
  {
    id: "43",
    label: "Shfaram",
  },
  {
    id: "44",
    label: "Nof Hagalil",
  },
  {
    id: "45",
    label: "Kiryat Yam",
  },
  {
    id: "46",
    label: "Kiryat Bialik",
  },
  {
    id: "47",
    label: "Kiryat Ono",
  },
  {
    id: "48",
    label: "Maale Adumim",
  },
  {
    id: "49",
    label: "Or Yehuda",
  },
  {
    id: "50",
    label: "Zefat",
  },
  {
    id: "51",
    label: "Netivot",
  },
  {
    id: "52",
    label: "Dimona",
  },
  {
    id: "53",
    label: "Tamra ",
  },
  {
    id: "54",
    label: "Sakhnin",
  },
  {
    id: "55",
    label: "Yehud-Monosson",
  },
  {
    id: "56",
    label: "Baka al-Gharbiya",
  },
  {
    id: "57",
    label: "Ofakim",
  },
  {
    id: "58",
    label: "Givat Shmuel",
  },
  {
    id: "59",
    label: "Tira",
  },
  {
    id: "60",
    label: "Arad",
  },
  {
    id: "61",
    label: "Migdal Haemek",
  },
  {
    id: "62",
    label: "Sderot",
  },
  {
    id: "63",
    label: "Araba",
  },
  {
    id: "64",
    label: "Nesher",
  },
  {
    id: "65",
    label: "Kiryat Shmona",
  },
  {
    id: "66",
    label: "Yokneam Illit",
  },
  {
    id: "67",
    label: "Kafr Qassem",
  },
  {
    id: "68",
    label: "Kfar Yona",
  },
  {
    id: "69",
    label: "Qalansawa",
  },
  {
    id: "70",
    label: "Kiryat Malachi",
  },
  {
    id: "71",
    label: "Maalot-Tarshiha",
  },
  {
    id: "72",
    label: "Tirat Carmel",
  },
  {
    id: "73",
    label: "Ariel",
  },
  {
    id: "74",
    label: "Or Akiva",
  },
  {
    id: "75",
    label: "Bet Shean",
  },
  {
    id: "76",
    label: "Mizpe Ramon",
  },
  {
    id: "77",
    label: "Lod",
  },
  {
    id: "78",
    label: "Nazareth",
  },
  {
    id: "79",
    label: "Qazrin",
  },
  {
    id: "80",
    label: "En Gedi",
  },
  {
    id: "81",
    label: "Ganei Tikva",
  },
  {
    id: "82",
    label: "Beer Yaakov",
  },
  {
    id: "83",
    label: "Maghar",
  },
  {
    id: "84",
    label: "Tel Aviv - Yafo",
  },
  {
    id: "200",
    label: "Nimrod Fortress",
  },
  {
    id: "201",
    label: "Banias",
  },
  {
    id: "202",
    label: "Tel Dan",
  },
  {
    id: "203",
    label: "Snir Stream",
  },
  {
    id: "204",
    label: "Horshat Tal ",
  },
  {
    id: "205",
    label: "Ayun Stream",
  },
  {
    id: "206",
    label: "Hula",
  },
  {
    id: "207",
    label: "Tel Hazor",
  },
  {
    id: "208",
    label: "Akhziv",
  },
  {
    id: "209",
    label: "Yehiam Fortress",
  },
  {
    id: "210",
    label: "Baram",
  },
  {
    id: "211",
    label: "Amud Stream",
  },
  {
    id: "212",
    label: "Korazim",
  },
  {
    id: "213",
    label: "Kfar Nahum",
  },
  {
    id: "214",
    label: "Majrase ",
  },
  {
    id: "215",
    label: "Meshushim Stream",
  },
  {
    id: "216",
    label: "Yehudiya ",
  },
  {
    id: "217",
    label: "Gamla",
  },
  {
    id: "218",
    label: "Kursi ",
  },
  {
    id: "219",
    label: "Hamat Tiberias",
  },
  {
    id: "220",
    label: "Arbel",
  },
  {
    id: "221",
    label: "En Afek",
  },
  {
    id: "222",
    label: "Tzipori",
  },
  {
    id: "223",
    label: "Hai-Bar Carmel",
  },
  {
    id: "224",
    label: "Mount Carmel",
  },
  {
    id: "225",
    label: "Bet Shearim",
  },
  {
    id: "226",
    label: "Mishmar HaCarmel ",
  },
  {
    id: "227",
    label: "Nahal Me‘arot",
  },
  {
    id: "228",
    label: "Dor-HaBonim",
  },
  {
    id: "229",
    label: "Tel Megiddo",
  },
  {
    id: "230",
    label: "Kokhav HaYarden",
  },
  {
    id: "231",
    label: "Maayan Harod",
  },
  {
    id: "232",
    label: "Bet Alpha",
  },
  {
    id: "233",
    label: "Gan HaShlosha",
  },
  {
    id: "235",
    label: "Taninim Stream",
  },
  {
    id: "236",
    label: "Caesarea",
  },
  {
    id: "237",
    label: "Tel Dor",
  },
  {
    id: "238",
    label: "Mikhmoret Sea Turtle",
  },
  {
    id: "239",
    label: "Beit Yanai",
  },
  {
    id: "240",
    label: "Apollonia",
  },
  {
    id: "241",
    label: "Mekorot HaYarkon",
  },
  {
    id: "242",
    label: "Palmahim",
  },
  {
    id: "243",
    label: "Castel",
  },
  {
    id: "244",
    label: "En Hemed",
  },
  {
    id: "245",
    label: "City of David",
  },
  {
    id: "246",
    label: "Me‘arat Soreq",
  },
  {
    id: "248",
    label: "Bet Guvrin",
  },
  {
    id: "249",
    label: "Sha’ar HaGai",
  },
  {
    id: "250",
    label: "Migdal Tsedek",
  },
  {
    id: "251",
    label: "Haniya Spring",
  },
  {
    id: "252",
    label: "Sebastia",
  },
  {
    id: "253",
    label: "Mount Gerizim",
  },
  {
    id: "254",
    label: "Nebi Samuel",
  },
  {
    id: "255",
    label: "En Prat",
  },
  {
    id: "256",
    label: "En Mabo‘a",
  },
  {
    id: "257",
    label: "Qasr al-Yahud",
  },
  {
    id: "258",
    label: "Good Samaritan",
  },
  {
    id: "259",
    label: "Euthymius Monastery",
  },
  {
    id: "261",
    label: "Qumran",
  },
  {
    id: "262",
    label: "Enot Tsukim",
  },
  {
    id: "263",
    label: "Herodium",
  },
  {
    id: "264",
    label: "Tel Hebron",
  },
  {
    id: "267",
    label: "Masada ",
  },
  {
    id: "268",
    label: "Tel Arad",
  },
  {
    id: "269",
    label: "Tel Beer Sheva",
  },
  {
    id: "270",
    label: "Eshkol",
  },
  {
    id: "271",
    label: "Mamshit",
  },
  {
    id: "272",
    label: "Shivta",
  },
  {
    id: "273",
    label: "Ben-Gurion’s Tomb",
  },
  {
    id: "274",
    label: "En Avdat",
  },
  {
    id: "275",
    label: "Avdat",
  },
  {
    id: "277",
    label: "Hay-Bar Yotvata",
  },
  {
    id: "278",
    label: "Coral Beach",
  },
  {
    id: "702",
    label: "Gilgal",
  },
  {
    id: "703",
    label: "Maale Gilboa",
  },
  {
    id: "704",
    label: "Makhtesh Ramon",
  },
  {
    id: "705",
    label: "Neot Semadar",
  },
  {
    id: "706",
    label: "Red Canyon",
  },
  {
    id: "707",
    label: "Hazeva",
  },
];

export default imsStations;
