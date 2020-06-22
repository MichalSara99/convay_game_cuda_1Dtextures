#include<iostream>
#include<string>


#include"convay_game.h"


int main(int argc, char const* argv[]) {


	std::size_t const boardSize = 20;
	std::size_t totalSize = boardSize * boardSize;
	std::size_t const gens = 100;


	ConvayGame game{ boardSize,gens };
	game.play();


	std::cin.get();
	std::cin.get();
	return 0;
}