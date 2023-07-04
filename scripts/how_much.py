# list of [prediction, result, odds] lists
half_list = [[0, 1.0, 165.0],
    [1, 1.0, 115.0],
    [0, 0.0, 100.0],
    [0, 1.0, 190.0],
    [1, 1.0, -120.0],
    [0, 0.0, 270.0],
    [0, 1.0, 165.0],
    [0, 0.0, 180.0],
    [1, 0.0, -230.0],
    [0, 1.0, 110.0],
    [0, 1.0, 145.0],
    [0, 0.0, 100.0],
    [0, 0.0, 210.0],
    [0, 1.0, 145.0],
    [0, 0.0, 135.0],
    [0, 1.0, 110.0],
    [0, 0.0, 140.0],
    [0, 0.0, 120.0],
    [0, 0.0, 195.0],
    [1, 0.0, -210.0],
    [0, 1.0, 190.0],
    [0, 0.0, 180.0],
    [0, 1.0, -110.0],
    [0, 0.0, -110.0],
    [1, 1.0, -190.0],
    [1, 1.0, -170.0],
    [0, 0.0, 145.0],
    [0, 0.0, 150.0],
    [0, 0.0, 153.0],
    [0, 0.0, 125.0],
    [0, 0.0, 155.0],
    [0, 0.0, 325.0],
    [0, 0.0, 110.0],
    [0, 1.0, 175.0],
    [1, 0.0, -160.0],
    [0, 0.0, 210.0],
    [1, 1.0, -166.0],
    [0, 0.0, -118.0],
    [0, 0.0, 195.0],
    [0, 0.0, 150.0],
    [1, 0.0, -160.0],
    [0, 0.0, 190.0],
    [1, 1.0, -125.0],
    [0, 0.0, 150.0],
    [0, 0.0, 135.0],
    [0, 0.0, 155.0],
    [0, 1.0, 195.0],
    [0, 0.0, 170.0],
    [1, 0.0, -115.0],
    [0, 0.0, 115.0],
    [0, 0.0, 110.0],
    [0, 0.0, 120.0],
    [0, 1.0, 121.0],
    [0, 0.0, 115.0],
    [0, 1.0, 152.0],
    [0, 0.0, 105.0],
    [1, 1.0, -170.0],
    [0, 1.0, 142.0],
    [1, 0.0, -115.0],
    [0, 1.0, -115.0],
    [0, 0.0, 220.0],
    [0, 1.0, 180.0],
    [0, 0.0, 105.0],
    [1, 1.0, -370.0],
    [0, 0.0, 120.0],
    [0, 0.0, 175.0],
    [0, 1.0, -115.0],
    [0, 0.0, 155.0],
    [1, 1.0, -200.0],
    [0, 1.0, 205.0],
    [0, 0.0, 135.0],
    [0, 1.0, 245.0],
    [0, 0.0, 245.0],
    [0, 0.0, 295.0],
    [0, 1.0, 165.0],
    [0, 0.0, 260.0],
    [0, 1.0, 125.0],
    [0, 0.0, 120.0],
    [0, 0.0, 200.0],
    [0, 0.0, -115.0],
    [0, 0.0, 135.0],
    [1, 0.0, -145.0],
    [0, 0.0, 220.0],
    [0, 0.0, 150.0],
    [0, 0.0, 150.0],
    [0, 0.0, 200.0],
    [0, 1.0, -105.0],
    [0, 0.0, 115.0],
    [0, 0.0, 157.0],
    [0, 1.0, 108.0],
    [0, 1.0, 110.0],
    [0, 0.0, 138.0],
    [0, 0.0, 220.0],
    [0, 0.0, 135.0],
    [1, 1.0, -145.0],
    [1, 1.0, -180.0],
    [0, 0.0, 220.0],
    [1, 1.0, -220.0],
    [1, 1.0, -190.0],
    [1, 1.0, -370.0],
    [0, 0.0, 200.0],
    [0, 0.0, 210.0],
    [0, 0.0, 165.0],
    [1, 0.0, -130.0],
    [0, 1.0, 135.0],
    [0, 0.0, 140.0],
    [0, 1.0, 155.0],
    [0, 0.0, 135.0],
    [0, 0.0, 135.0],
    [0, 1.0, 115.0],
    [0, 0.0, 145.0],
    [0, 0.0, 100.0],
    [0, 1.0, 145.0],
    [0, 0.0, 125.0],
    [0, 0.0, 200.0],
    [0, 1.0, -130.0],
    [0, 0.0, 165.0],
    [0, 1.0, 105.0],
    [1, 1.0, 151.0],
    [0, 0.0, 185.0],
    [0, 0.0, 205.0],
    [0, 0.0, 125.0],
    [0, 1.0, 240.0],
    [0, 1.0, 110.0],
    [1, 0.0, -170.0],
    [0, 0.0, 175.0],
    [0, 0.0, 180.0],
    [0, 0.0, 165.0],
    [0, 0.0, 165.0],
    [0, 0.0, 155.0],
    [0, 0.0, 135.0],
    [0, 0.0, 165.0],
    [0, 1.0, 225.0],
    [0, 0.0, -135.0],
    [0, 0.0, 135.0],
    [0, 0.0, 185.0],
    [0, 0.0, 170.0],
    [0, 0.0, 220.0],
    [0, 1.0, 180.0],
    [0, 1.0, -110.0],
    [0, 1.0, 245.0],
    [0, 1.0, 140.0],
    [0, 0.0, 102.0],
    [0, 0.0, 170.0],
    [0, 0.0, -105.0],
    [0, 1.0, 135.0],
    [0, 1.0, 175.0],
    [1, 0.0, -160.0],
    [0, 1.0, 130.0],
    [1, 0.0, 135.0],
    [0, 0.0, 165.0],
    [1, 1.0, -128.0],
    [0, 0.0, 200.0],
    [0, 0.0, 155.0],
    [1, 1.0, -120.0],
    [0, 1.0, 174.0],
    [1, 1.0, -240.0],
    [0, 1.0, 250.0],
    [0, 0.0, 255.0],
    [0, 0.0, 202.0],
    [0, 0.0, -135.0],
    [0, 0.0, -195.0],
    [1, 1.0, -145.0],
    [0, 1.0, -162.0],
    [0, 0.0, 120.0],
    [0, 1.0, -185.0],
    [0, 0.0, 160.0],
    [0, 1.0, 116.0],
    [1, 1.0, -120.0],
    [0, 0.0, 225.0],
    [0, 1.0, 155.0],
    [0, 1.0, 110.0],
    [1, 1.0, -180.0],
    [0, 0.0, 130.0],
    [0, 0.0, 125.0],
    [0, 1.0, -120.0],
    [0, 0.0, 195.0],
    [0, 1.0, 240.0],
    [0, 1.0, 425.0],
    [0, 1.0, 175.0],
    [1, 1.0, -140.0],
    [0, 0.0, 135.0],
    [0, 0.0, 135.0],
    [0, 0.0, 110.0],
    [1, 1.0, -150.0],
    [0, 0.0, 205.0],
    [0, 0.0, -115.0],
    [0, 1.0, 225.0],
    [0, 0.0, 255.0],
    [0, 0.0, 235.0],
    [0, 0.0, 135.0],
    [0, 0.0, 120.0],
    [0, 1.0, 180.0],
    [0, 0.0, 155.0],
    [1, 0.0, 151.0],
    [0, 0.0, 180.0],
    [0, 0.0, 115.0],
    [0, 0.0, 180.0],
    [0, 0.0, 170.0],
    [0, 1.0, 105.0],
    [1, 0.0, -145.0],
    [0, 0.0, 100.0],
    [0, 0.0, 135.0],
    [0, 0.0, 125.0],
    [0, 0.0, 155.0],
    [0, 0.0, 170.0],
    [0, 0.0, 185.0],
    [0, 0.0, 135.0],
    [1, 1.0, -195.0],
    [0, 1.0, -110.0],
    [0, 1.0, 235.0],
    [0, 0.0, -166.0],
    [1, 1.0, -170.0],
    [0, 0.0, 120.0],
    [0, 0.0, -170.0],
    [0, 1.0, 150.0],
    [0, 0.0, 150.0],
    [0, 0.0, 150.0],
    [0, 0.0, 175.0],
    [0, 0.0, 130.0]]


# formula to calculate how much you win given american odds
def calculate_winnings(odds, bet_amount):
    if odds > 0:
        profit = (odds / 100) * bet_amount
        return profit
    elif odds < 0:
        profit = (100 / abs(odds)) * bet_amount
        return profit
    else:
        return 0


total_winnings = 0
for item in half_list:
    if int(item[0]) == 1 and int(item[1]) == 1:
        winnings = calculate_winnings(item[2], 100)
        total_winnings += winnings
    elif int(item[0]) == 1 and int(item[1]) != 1:
        losings = calculate_winnings(item[2], 100)
        total_winnings -= losings

print(total_winnings * 9) # the 9 comes from the half_list only accounting for 1/9 of the seasons games